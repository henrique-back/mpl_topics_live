"""Consensus clustering of the topic model over many UMAP seeds (evidence accumulation, Fred & Jain 2005).

A single UMAP + HDBSCAN fit depends on its seed: two refits of the same parameters agree only partly (ARI 0.54). The
consensus runs the fit with many seeds and counts, for every pair of papers, the share of runs in which they fall in
the same cluster (the co-association). Average linkage on 1 - co-association, cut at 1 - `threshold`, gives the
topics: groups of papers that are, on average, clustered together in at least `threshold` of the runs. Groups smaller
than `min_size` papers are outliers (-1).

The linkage (tree) is computed on the papers clustered in at least `threshold` of the runs: the others can never be
merged below the cut, as their co-association with any paper is at most their own. The same holds for any higher
threshold, so one tree serves every threshold at least as high (cut). The condensed distance matrix (4 n^2 bytes for
the n papers kept) is built without a dense n x n copy. On 20,000 papers with synthetic, noisy runs, partition() peaked
at 4.9 GB, most of it the sparse co-association, against 5.8 GB with a dense copy.

ConsensusCluster wraps the partition as a BERTopic cluster model: it predicts the topic of a new paper by a vote of
its nearest papers in the embedding space, so assign.py can place new papers with BERTopic's transform().
"""
import numpy as np
import scipy.sparse as sp
from hdbscan import HDBSCAN
from joblib import Parallel, delayed
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, cohen_kappa_score
from sklearn.neighbors import NearestNeighbors
from umap import UMAP

import config


def indicator(labels):
    """Sparse paper x cluster membership matrix (outliers have no column) and the cluster ids of its columns."""
    labels = np.asarray(labels)
    ids, columns = np.unique(labels[labels != -1], return_inverse=True)
    rows = np.flatnonzero(labels != -1)
    m = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, columns)), shape=(len(labels), len(ids)))
    return m, ids


def jaccard(a, b):
    """Jaccard index of the papers of every cluster of `a` with those of every cluster of `b` (outliers excluded).

    Returns (cluster ids of a, cluster ids of b, matrix)."""
    ma, ids_a = indicator(a)
    mb, ids_b = indicator(b)
    shared = (ma.T @ mb).toarray()
    size_a, size_b = np.asarray(ma.sum(axis=0)).ravel(), np.asarray(mb.sum(axis=0)).ravel()
    return ids_a, ids_b, shared / (size_a[:, None] + size_b[None, :] - shared)


def agreement(a, b):
    """Agreement of two labelings of the same papers (-1 = outlier): ARI (outliers as one cluster), AMI on the papers
    clustered in both, Cohen's kappa on outlier status, and the mean over the topics of both of the Jaccard index with
    their best match in the other."""
    a, b = np.asarray(a), np.asarray(b)
    both = (a != -1) & (b != -1)
    _, _, j = jaccard(a, b)
    best = np.concatenate([j.max(axis=1), j.max(axis=0)]) if j.size else np.array([])
    return dict(ari=float(adjusted_rand_score(a, b)),
                ami=float(adjusted_mutual_info_score(a[both], b[both])) if both.sum() > 1 else np.nan,
                kappa=float(cohen_kappa_score(a == -1, b == -1)),
                jaccard=float(best.mean()) if len(best) else np.nan)


def run(embeddings, umap_params, hdbscan_params, seed):
    """Topic labels of one UMAP + HDBSCAN fit with the UMAP seed `seed`."""
    reduced = UMAP(**{**umap_params, 'random_state': seed}).fit_transform(embeddings)
    return cluster(reduced, hdbscan_params)


def cluster(reduced, hdbscan_params):
    """HDBSCAN labels of one UMAP embedding. Single-threaded (the runs are parallel) and without prediction data,
    which the consensus does not use."""
    return HDBSCAN(**{**hdbscan_params, 'prediction_data': False, 'core_dist_n_jobs': 1}).fit(reduced).labels_


def runs(embeddings, umap_params, hdbscan_params, seeds, n_jobs=-1):
    """Labels of one fit per seed, in parallel (UMAP is single-threaded when seeded)."""
    return Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(run)(embeddings, umap_params, hdbscan_params, seed) for seed in seeds)


def coassociation(label_runs):
    """Sparse matrix of the share of runs in which each pair of papers is in the same (non-outlier) cluster.

    Its diagonal is the share of runs in which each paper is clustered at all."""
    total = None
    for labels in label_runs:
        m, _ = indicator(labels)
        pairs = m @ m.T
        total = pairs if total is None else total + pairs
    return (total / len(label_runs)).tocsr()


BLOCK = 2000  # rows of the co-association densified at a time


def tree(c, threshold=0.5):
    """Average-linkage tree on 1 - co-association of the papers clustered in at least `threshold` of the runs:
    (linkage matrix or None, indices of those papers)."""
    keep = np.flatnonzero(c.diagonal() >= threshold)
    if len(keep) < 2:
        return None, keep
    sub = c[keep][:, keep].tocsr()
    m = len(keep)
    condensed = np.empty(m * (m - 1) // 2)
    start = 0
    for i0 in range(0, m, BLOCK):
        block = sub[i0:i0 + BLOCK].toarray()
        for r in range(len(block)):
            row = block[r, i0 + r + 1:]
            condensed[start:start + len(row)] = 1 - row
            start += len(row)
    del sub
    return linkage(condensed, method='average'), keep


def cut(c, linkage_tree, keep, threshold=0.5, min_size=10):
    """Topics of the tree cut at 1 - `threshold` (`threshold` at least that of the tree): groups of at least
    `min_size` papers, numbered by decreasing size; the rest, and the papers clustered in fewer than `threshold` of
    the runs, are outliers (-1)."""
    labels = np.full(c.shape[0], -1)
    if linkage_tree is None:
        return labels
    groups = fcluster(linkage_tree, t=1 - threshold, criterion='distance')
    groups[c.diagonal()[keep] < threshold] = 0  # never merged below the cut; excluded explicitly for min_size 1
    ids, sizes = np.unique(groups[groups > 0], return_counts=True)
    big = ids[sizes >= min_size][np.argsort(-sizes[sizes >= min_size], kind='stable')]
    rank = {group: topic for topic, group in enumerate(big)}
    labels[keep] = [rank.get(group, -1) for group in groups]
    return labels


def confidence(c, labels):
    """Mean co-association of each paper with the other papers of its topic (NaN for outliers)."""
    labels = np.asarray(labels)
    result = np.full(len(labels), np.nan)
    clustered = np.flatnonzero(labels != -1)
    if len(clustered):
        m, _ = indicator(labels)  # topics are 0..k-1, so a topic is its own column
        topic = labels[clustered]
        within = np.asarray((c @ m).tocsr()[clustered, topic]).ravel() - c.diagonal()[clustered]
        sizes = np.asarray(m.sum(axis=0)).ravel()[topic]
        result[clustered] = within / np.maximum(sizes - 1, 1)
    return result


def partition(c, threshold=0.5, min_size=10):
    """Consensus topics from the co-association `c`: (labels, confidence). See tree() and cut()."""
    labels = cut(c, *tree(c, threshold), threshold, min_size)
    return labels, confidence(c, labels)


def vote(neighbour_labels):
    """Most common label of each row of `neighbour_labels` (papers x neighbours); ties go to the lowest label, so to
    -1 when it is among the most common."""
    neighbour_labels = np.asarray(neighbour_labels)
    shift = neighbour_labels + 1  # -1 -> 0
    counts = np.zeros((len(shift), shift.max() + 1 if shift.size else 1), dtype=int)
    np.add.at(counts, (np.repeat(np.arange(len(shift)), shift.shape[1]), shift.ravel()), 1)
    return counts.argmax(axis=1) - 1  # argmax takes the first of the most common


class ConsensusCluster:
    """BERTopic cluster model of a fixed (consensus) partition of the papers `paper_ids`.

    fit() takes the topics as `y`. predict() gives a paper the most common topic among its `n_neighbors` nearest
    training papers (cosine in the embedding space), with the outliers voting as topic -1. Ties go to -1 when it is
    among the most common, otherwise to the lowest topic id (the largest topic). The training embeddings are read from
    embeddings.npz when first needed, not pickled with the model, and must have been made by the encoder of the fit
    (config.encoder() then). Use it with bertopic.dimensionality.BaseDimensionalityReduction, so predict() receives
    the embeddings."""

    def __init__(self, paper_ids, n_neighbors=15):
        self.paper_ids = np.asarray(paper_ids, dtype=str)
        self.n_neighbors = n_neighbors
        self.encoder = config.encoder()

    def fit(self, X, y=None):
        if y is None or len(y) != len(self.paper_ids):
            raise ValueError("ConsensusCluster.fit needs the topic of every training paper as y")
        self.labels_ = np.asarray(y)
        self._index = None
        return self

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_index'] = None
        return state

    def _neighbours(self):
        if getattr(self, '_index', None) is None:
            self._index = NearestNeighbors(n_neighbors=self.n_neighbors, metric='cosine', algorithm='brute').fit(
                config.load_embeddings(self.paper_ids, expect=self.encoder))
        return self._index

    def _vote(self, neighbours):
        return vote(self.labels_[neighbours])

    def predict(self, X):
        return self._vote(self._neighbours().kneighbors(np.asarray(X), return_distance=False))

    def loo_predict(self):
        """The topic each training paper gets from the vote of its `n_neighbors` nearest other training papers: how
        predict() treats new papers like them."""
        return self._vote(self._neighbours().kneighbors(n_neighbors=self.n_neighbors, return_distance=False))
