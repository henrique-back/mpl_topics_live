"""Choose the topic model's parameters by evaluation. Run the stages in order; each writes to bertopic/evaluation/.

  sizes   The topic size each analysis needs, independent of any model: the precision of a topic's Gini coefficient
          (against its number of papers old enough for the citation analyses) and basin shares, and its papers per
          year, against the number of papers in the topic (random subsets of the corpus). Read off a minimum size per
          analysis and set config.TOPIC_LEVELS and config.MIN_GINI_PAPERS by hand.
  umap    The UMAP representation, chosen apart from the clustering: neighbourhood preservation (trustworthiness and
          kNN recall against the embedding space) and seed stability, over a grid of n_neighbors x n_components.
          Rule: take the most seed-stable setting whose kNN recall is within UMAP_TOLERANCE of the best; set
          config.UMAP_PARAMS by hand.
  search  The HDBSCAN grid on the fixed UMAP of config.UMAP_PARAMS, over SEEDS UMAP seeds (plus held-out folds):
          A noise-penalized DBCV (relative_validity_, ranked within each seed), B NPMI coherence and topic diversity,
          C stability across seeds, D held-out assignment, E author-keyword information.
  report  The Pareto front over stability, DBCV, keyword information and the held-out outlier rate.
  consensus  The consensus that baseline.py fits (consensus.py), for the candidates of the Pareto front (or
          --configs): for each CONSENSUS_THRESHOLDS, the topics, outliers and split-half agreement of a consensus of
          config.CONSENSUS_RUNS runs, and for each CONSENSUS_KNNS the leave-one-out vote of the papers (the share that
          keep their topic, and the share left unmatched, as new papers would be). The vote is scored against a
          consensus that saw those papers, so it tests the assignment rule, not how well the topics generalize.
          --converge adds, for the candidates, the split-half agreement against the number of runs per half (set
          config.CONSENSUS_RUNS). With --configs, the results go to consensus_selected.* instead of consensus.*. Choose the fine level from it, set config.HDBSCAN_PARAMS and the CONSENSUS_*
          settings by hand and record the chosen row and why in bertopic/evaluation/CHOICE.md.

UMAP embeddings are cached in bertopic/evaluation/cache/ (not committed), keyed on the papers, the encoder and the UMAP
parameters, so an interrupted stage resumes. --sample N runs on a random subset of N papers and --quick on reduced
grids, for checks.
"""
import argparse
import hashlib
import itertools
import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.sparse as sp
from bertopic.vectorizers import ClassTfidfTransformer
from hdbscan import HDBSCAN, approximate_predict
from joblib import Parallel, delayed
from plotly.subplots import make_subplots
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import trustworthiness
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from umap import UMAP

import config
import consensus
from analysis import CITES, citation_rates, gini

OUT = config.EVALUATION_DIR
CACHE = OUT / 'cache'

# sizes
SIZE_GRID = [25, 50, 75, 100, 150, 200, 300, 400, 600, 800, 1000]
N_BOOT = 1000
N_BASINS = 5        # basin shares reported for the largest basins
N_YEARS = 10        # papers per year over the last full years, up to config.last_full_year()

# umap
UMAP_GRID = dict(n_neighbors=[5, 10, 15, 30, 50], n_components=[5, 10, 15, 25])
UMAP_SEEDS = [0, 1, 2]
K = 15              # neighbourhood size of trustworthiness and kNN recall
TRUST_SAMPLE = 5000
UMAP_TOLERANCE = 0.05  # relative to the best kNN recall

# search
SEEDS = list(range(10))
FOLD_SEEDS = [0, 1, 2]
N_FOLDS = 5
HDBSCAN_GRID = dict(min_cluster_size=[10, 15, 20, 30, 40, 50, 75], min_samples=[1, 2, 5, 10],
                    cluster_selection_epsilon=[0.0, 0.15])
TOP_WORDS = 10
MIN_KEYWORD_PAPERS = 20
N_PERMUTATIONS = 3
STABLE_JACCARD = 0.75  # a topic is stable if its best match has at least this Jaccard index ...
STABLE_SHARE = 0.8     # ... in at least this share of the other seeds' runs

# report: objective -> True to maximize
OBJECTIVES = {'stability_ami': True, 'dbcv_score': True, 'keyword_score': True, 'heldout_outliers': False}

# consensus
CONSENSUS_THRESHOLDS = [0.4, 0.5, 0.6, 0.7]
CONSENSUS_KNNS = [5, 10, 15, 25, 50]
CONVERGENCE_HALVES = [5, 10, 15, 20]  # runs per half


def quick():
    """Reduced grids, for checking the stages end to end."""
    global SIZE_GRID, N_BOOT, UMAP_GRID, UMAP_SEEDS, SEEDS, FOLD_SEEDS, N_FOLDS, HDBSCAN_GRID
    global CONSENSUS_THRESHOLDS, CONSENSUS_KNNS, CONVERGENCE_HALVES
    SIZE_GRID, N_BOOT = [25, 100, 400], 50
    UMAP_GRID, UMAP_SEEDS = dict(n_neighbors=[10, 30], n_components=[5, 10]), [0, 1]
    SEEDS, FOLD_SEEDS, N_FOLDS = [0, 1, 2], [0], 3
    HDBSCAN_GRID = dict(min_cluster_size=[15, 40], min_samples=[2, 10], cluster_selection_epsilon=[0.0])
    CONSENSUS_THRESHOLDS, CONSENSUS_KNNS, CONVERGENCE_HALVES = [0.5, 0.7], [5, 15], [2, 3]


def load(sample=None):
    """(papers, their embeddings, a key of the papers and encoder for the cache)."""
    data = pd.read_csv(config.ALLDATA)
    data['paper_id'] = config.paper_ids(data)
    if sample:
        data = data.sample(sample, random_state=0).reset_index(drop=True)
    key = hashlib.sha1('\n'.join([config.encoder(), *data['paper_id']]).encode()).hexdigest()[:10]
    return data, config.load_embeddings(data['paper_id']), key


# ---------------------------------------------------------------------------------------------
# UMAP embeddings (cached)
# ---------------------------------------------------------------------------------------------

def umap_embedding(embeddings, key, n_neighbors, n_components, seed, fold=None):
    """UMAP of all papers with config.UMAP_PARAMS but `n_neighbors`, `n_components` and the seed `seed`. With `fold`
    = (k, train mask), it is fitted on the training papers and the others are placed with transform(), as new papers
    are. Cached under `key` (load()) and the parameters; written atomically, so an interrupted run leaves no partial
    file."""
    params = {**config.UMAP_PARAMS, 'n_neighbors': n_neighbors, 'n_components': n_components, 'random_state': seed}
    tag = '' if fold is None else f'_fold{fold[0]}'
    other = hashlib.sha1(json.dumps({k: v for k, v in params.items()
                                     if k not in ('n_neighbors', 'n_components', 'random_state')},
                                    sort_keys=True).encode()).hexdigest()[:6]
    path = CACHE / f'umap_{key}_{other}_nn{n_neighbors}_nc{n_components}_seed{seed}{tag}.npy'
    if path.exists():
        return np.load(path)
    model = UMAP(**params)
    if fold is None:
        reduced = model.fit_transform(embeddings)
    else:
        train = fold[1]
        reduced = np.empty((len(embeddings), n_components), dtype=np.float32)
        reduced[train] = model.fit_transform(embeddings[train])
        reduced[~train] = model.transform(embeddings[~train])
    reduced = reduced.astype(np.float32)
    CACHE.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.stem + '.partial.npy')
    np.save(partial, reduced)
    os.replace(partial, path)
    return reduced


def folds(n, seed):
    """Training masks of the N_FOLDS folds of seed `seed`."""
    masks = []
    for train, _ in KFold(N_FOLDS, shuffle=True, random_state=seed).split(np.arange(n)):
        mask = np.zeros(n, dtype=bool)
        mask[train] = True
        masks.append(mask)
    return masks


# ---------------------------------------------------------------------------------------------
# Stage sizes
# ---------------------------------------------------------------------------------------------

def stage_sizes(data):
    rng = np.random.default_rng(0)
    cites = citation_rates(data)[CITES].dropna().to_numpy()  # the papers old enough for the citation analyses
    basins = data['paper_id'].map(pd.read_csv(config.BASIN_LABELS).set_index('paper_id')['Basin'])
    major = basins.value_counts().index[:N_BASINS]
    basin_codes = basins.map({b: i for i, b in enumerate(major)}).fillna(-1).to_numpy(int)  # -1: other or none
    located = basins.notna().to_numpy()
    years = data['Year'].to_numpy()
    last = np.arange(config.last_full_year() - N_YEARS + 1, config.last_full_year() + 1)
    year_share = np.array([(years == y).mean() for y in last])

    rows = []
    for n in SIZE_GRID:
        if n > len(data):
            break
        ginis, shares, n_located = [], [], []
        for _ in range(N_BOOT):
            ginis.append(gini(rng.choice(cites, n, replace=False)) if n <= len(cites) else np.nan)
            idx = rng.choice(len(data), n, replace=False)
            loc = idx[located[idx]]
            n_located.append(len(loc))
            shares.append([np.mean(basin_codes[loc] == b) if len(loc) else np.nan for b in range(len(major))])
        shares = np.array(shares)
        row = dict(n=n, gini_mean=np.mean(ginis), gini_halfwidth=np.subtract(*np.percentile(ginis, [97.5, 2.5])) / 2,
                   located_papers=np.mean(n_located))
        for b, name in enumerate(major):
            row[f'basin_halfwidth_{name}'] = np.subtract(*np.nanpercentile(shares[:, b], [97.5, 2.5])) / 2
        row['basin_halfwidth_max'] = max(row[f'basin_halfwidth_{name}'] for name in major)
        row['papers_per_year_median'] = n * np.median(year_share)
        row['papers_per_year_min'] = n * year_share.min()
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / 'sizes.csv', index=False)

    panels = {'Gini: 95% CI half-width': ['gini_halfwidth'],
              'Basin share: 95% CI half-width': [c for c in table if c.startswith('basin_halfwidth_')],
              f'Papers per year ({last[0]}-{last[-1]})': ['papers_per_year_median', 'papers_per_year_min']}
    fig = make_subplots(rows=len(panels), cols=1, subplot_titles=list(panels), shared_xaxes=True)
    for i, (title, columns) in enumerate(panels.items(), 1):
        for column in columns:
            fig.add_trace(go.Scatter(x=table['n'], y=table[column], mode='lines+markers',
                                     name=column.replace('basin_halfwidth_', '')), row=i, col=1)
    fig.update_xaxes(type='log', title_text='Papers in the topic', row=len(panels), col=1)
    fig.update_layout(height=900, title='Precision of the analyses against topic size')
    fig.write_html(OUT / 'sizes.html', include_plotlyjs='cdn')
    print(table.round(3).to_string(index=False))


# ---------------------------------------------------------------------------------------------
# Stage umap
# ---------------------------------------------------------------------------------------------

def knn(x, metric):
    """Indices of the K nearest neighbours of each point (itself excluded)."""
    algorithm = 'brute' if metric == 'cosine' else 'auto'
    index = NearestNeighbors(n_neighbors=K + 1, metric=metric, algorithm=algorithm).fit(x)
    return index.kneighbors(x, return_distance=False)[:, 1:]


def overlap(a, b):
    """Mean share of shared neighbours between two kNN index arrays."""
    return float(np.mean([len(set(i) & set(j)) / K for i, j in zip(a, b)]))


def stage_umap(embeddings, key):
    grid = list(itertools.product(UMAP_GRID['n_neighbors'], UMAP_GRID['n_components'], UMAP_SEEDS))
    reduced = Parallel(n_jobs=-1, verbose=5)(
        delayed(umap_embedding)(embeddings, key, nn, nc, seed) for nn, nc, seed in grid)
    high = knn(embeddings, 'cosine')
    sample = np.random.default_rng(0).choice(len(embeddings), min(TRUST_SAMPLE, len(embeddings)), replace=False)

    rows, neighbours = [], {}
    for (nn, nc, seed), x in zip(grid, reduced):
        low = knn(x, 'euclidean')
        neighbours[nn, nc, seed] = low
        rows.append(dict(n_neighbors=nn, n_components=nc, seed=seed, knn_recall=overlap(high, low),
                         trustworthiness=trustworthiness(embeddings[sample], x[sample], n_neighbors=K,
                                                         metric='cosine')))
    runs = pd.DataFrame(rows)
    table = runs.groupby(['n_neighbors', 'n_components'])[['knn_recall', 'trustworthiness']].agg(['mean', 'std'])
    table.columns = ['_'.join(c) for c in table.columns]
    table['seed_stability'] = [np.mean([overlap(neighbours[nn, nc, a], neighbours[nn, nc, b])
                                        for a, b in itertools.combinations(UMAP_SEEDS, 2)])
                               for nn, nc in table.index]
    table = table.reset_index()
    table.to_csv(OUT / 'umap.csv', index=False)

    fig = make_subplots(rows=1, cols=3, subplot_titles=['kNN recall', 'Trustworthiness', 'Seed stability'])
    for i, column in enumerate(['knn_recall_mean', 'trustworthiness_mean', 'seed_stability'], 1):
        for nn, group in table.groupby('n_neighbors'):
            fig.add_trace(go.Scatter(x=group['n_components'], y=group[column], mode='lines+markers',
                                     name=f'n_neighbors={nn}', legendgroup=str(nn), showlegend=i == 1), row=1, col=i)
        fig.update_xaxes(title_text='n_components', row=1, col=i)
    fig.update_layout(height=450, title='UMAP neighbourhood preservation (against the cosine kNN of the embeddings)')
    fig.write_html(OUT / 'umap.html', include_plotlyjs='cdn')

    print(table.round(3).to_string(index=False))
    # Neighbourhood preservation and seed stability pull apart (small n_neighbors keeps the closest neighbours best
    # but varies most across seeds): the front of the two, and the most seed-stable setting near the best recall
    values = table[['knn_recall_mean', 'seed_stability']].to_numpy()
    front = [not ((values >= v).all(axis=1) & (values > v).any(axis=1)).any() for v in values]
    print("\nPareto front of kNN recall and seed stability:")
    print(table.loc[front].round(3).to_string(index=False))
    best = table['knn_recall_mean'].max()
    pick = table[table['knn_recall_mean'] >= (1 - UMAP_TOLERANCE) * best].sort_values(
        ['seed_stability', 'n_components'], ascending=[False, True]).iloc[0]
    print(f"\nSuggested (most seed-stable within {UMAP_TOLERANCE:.0%} of the best kNN recall): "
          f"n_neighbors={int(pick['n_neighbors'])}, n_components={int(pick['n_components'])}")


# ---------------------------------------------------------------------------------------------
# Stage search: metrics
# ---------------------------------------------------------------------------------------------

def doc_term(docs):
    """Document x term counts, as in the model's vectorizer but with min_df=5."""
    vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words='english', min_df=5)
    return vectorizer.fit_transform(docs).tocsc()


def keyword_matrix(data):
    """Paper x author-keyword presence, for the keywords of at least MIN_KEYWORD_PAPERS papers."""
    keywords = (data['Author Keywords'].fillna('').str.lower().str.split(';')
                .apply(lambda ks: sorted({k.strip() for k in ks if k.strip()})))
    counts = keywords.explode().value_counts()
    kept = {k: i for i, k in enumerate(counts.index[counts >= MIN_KEYWORD_PAPERS])}
    rows, cols = zip(*[(r, kept[k]) for r, ks in enumerate(keywords) for k in ks if k in kept]) or ((), ())
    return sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(data), len(kept)))


def words(x, labels):
    """Top TOP_WORDS c-TF-IDF terms of each topic (BERTopic's c-TF-IDF, outliers included in the idf)."""
    with_outliers = np.where(labels == -1, labels.max() + 1, labels)
    m, ids = consensus.indicator(with_outliers)
    ctfidf = ClassTfidfTransformer().fit_transform((m.T @ x).tocsr()).tocsr()
    top = []
    for row, topic in enumerate(ids):
        if topic == labels.max() + 1:
            continue
        start, end = ctfidf.indptr[row], ctfidf.indptr[row + 1]
        order = np.argsort(-ctfidf.data[start:end])[:TOP_WORDS]
        top.append(ctfidf.indices[start:end][order])
    return top


def npmi(binary, top):
    """Mean NPMI of the pairs of top words of each topic, over the topics (document co-occurrence in the corpus)."""
    n = binary.shape[0]
    scores = []
    for terms in top:
        b = binary[:, terms].toarray().astype(np.float32)
        joint = (b.T @ b) / n
        p = np.diag(joint)
        i, j = np.triu_indices(len(terms), 1)
        pij = joint[i, j]
        with np.errstate(divide='ignore', invalid='ignore'):
            value = np.where(pij > 0, np.log(pij / (p[i] * p[j])) / -np.log(pij), -1.0)
        scores.append(value.mean())
    return float(np.mean(scores)) if scores else np.nan


def mutual_information(labels, keywords):
    """MI between the topic labels (-1 is a label) and the presence of each keyword."""
    m, _ = consensus.indicator(labels + 1)
    n = len(labels)
    joint1 = (m.T @ keywords).toarray() / n                 # p(topic, keyword present)
    p_topic = np.asarray(m.sum(axis=0)).ravel()[:, None] / n
    p_kw = joint1.sum(axis=0)[None, :]
    joint0 = p_topic - joint1                               # p(topic, keyword absent)
    with np.errstate(divide='ignore', invalid='ignore'):
        mi = np.nansum(np.where(joint1 > 0, joint1 * np.log(joint1 / (p_topic * p_kw)), 0)
                       + np.where(joint0 > 0, joint0 * np.log(joint0 / (p_topic * (1 - p_kw))), 0), axis=0)
    return mi, p_kw.ravel()


def keyword_score(labels, keywords, seed=0):
    """Frequency-weighted mean over the keywords of (MI - MI of permuted labels) / H(keyword)."""
    if keywords.shape[1] == 0:
        return np.nan
    rng = np.random.default_rng(seed)
    mi, p = mutual_information(labels, keywords)
    chance = np.mean([mutual_information(rng.permutation(labels), keywords)[0] for _ in range(N_PERMUTATIONS)],
                     axis=0)
    entropy = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    return float(np.average((mi - chance) / entropy, weights=p))


def fit_config(reduced, params, train=None):
    """HDBSCAN on `reduced`: (labels, relative validity), or for a fold the held-out predictions."""
    model = HDBSCAN(**params, metric='euclidean', cluster_selection_method='eom', core_dist_n_jobs=1,
                    gen_min_span_tree=train is None, prediction_data=train is not None)
    if train is None:
        model.fit(reduced)
        try:
            validity = float(model.relative_validity_)
        except Exception:
            validity = np.nan
        return model.labels_, validity
    model.fit(reduced[train])
    return approximate_predict(model, reduced[~train])[0], np.nan


def run_metrics(labels, validity, x, binary, keywords):
    clustered = labels[labels != -1]
    sizes = np.bincount(clustered) if len(clustered) else np.array([0])
    sizes = sizes[sizes > 0]
    top = words(x, labels) if len(clustered) else []
    n_terms = sum(len(t) for t in top)
    return dict(n_topics=len(sizes), noise_frac=float(np.mean(labels == -1)), dbcv=validity,
                size_min=int(sizes.min()), size_median=float(np.median(sizes)), size_max=int(sizes.max()),
                npmi=npmi(binary, top), diversity=len(set(np.concatenate(top))) / n_terms if n_terms else np.nan,
                keyword_score=keyword_score(labels, keywords))


def full_task(reduced, params, x, binary, keywords):
    labels, validity = fit_config(reduced, params)
    return labels, run_metrics(labels, validity, x, binary, keywords)


def stability(label_runs):
    """AMI on the papers clustered in both runs, kappa on outlier status, and best-match Jaccard of the topics."""
    ami, kappa = [], []
    best = {}  # (run, topic) -> best-match Jaccard in each other run
    for a, b in itertools.combinations(range(len(label_runs)), 2):
        la, lb = label_runs[a], label_runs[b]
        pair = consensus.agreement(la, lb)
        ami.append(pair['ami'])
        kappa.append(pair['kappa'])
        ids_a, ids_b, j = consensus.jaccard(la, lb)
        if j.size:
            for topic, value in zip(ids_a, j.max(axis=1)):
                best.setdefault((a, topic), []).append(value)
            for topic, value in zip(ids_b, j.max(axis=0)):
                best.setdefault((b, topic), []).append(value)
    values = list(best.values())
    return dict(stability_ami=np.nanmean(ami), stability_kappa=np.mean(kappa),
                stability_jaccard=np.mean([np.mean(v) for v in values]) if values else np.nan,
                stable_topics=np.mean([np.mean(np.array(v) >= STABLE_JACCARD) >= STABLE_SHARE for v in values])
                if values else np.nan)


def stage_search(data, embeddings, key):
    nn, nc = config.UMAP_PARAMS['n_neighbors'], config.UMAP_PARAMS['n_components']
    n = len(embeddings)
    print(f"UMAP n_neighbors={nn}, n_components={nc}; {len(SEEDS)} seeds, {len(FOLD_SEEDS)} x {N_FOLDS} folds")
    fold_masks = {s: folds(n, s) for s in FOLD_SEEDS}
    umap_tasks = [(s, None) for s in SEEDS] + [(s, (k, m)) for s in FOLD_SEEDS for k, m in enumerate(fold_masks[s])]
    reduced = Parallel(n_jobs=-1, verbose=5)(
        delayed(umap_embedding)(embeddings, key, nn, nc, s, fold) for s, fold in umap_tasks)
    full = dict(zip(SEEDS, reduced[:len(SEEDS)]))
    folded = dict(zip([(s, fold[0]) for s, fold in umap_tasks[len(SEEDS):]], reduced[len(SEEDS):]))

    configs = [dict(zip(HDBSCAN_GRID, values)) for values in itertools.product(*HDBSCAN_GRID.values())]
    x = doc_term(data['Abstract'].tolist())
    binary = (x > 0).astype(np.float32).tocsc()
    keywords = keyword_matrix(data)
    print(f"{len(configs)} HDBSCAN configurations; {x.shape[1]} terms, {keywords.shape[1]} keywords")

    tasks = [(c, s) for c in range(len(configs)) for s in SEEDS]
    results = Parallel(n_jobs=-1, verbose=5)(
        delayed(full_task)(full[s], configs[c], x, binary, keywords) for c, s in tasks)
    labels = {task: result[0] for task, result in zip(tasks, results)}
    runs = pd.DataFrame([{'config': c, **configs[c], 'seed': s, **result[1]} for (c, s), result in zip(tasks, results)])
    runs['dbcv_score'] = runs.groupby('seed')['dbcv'].rank(pct=True)

    fold_tasks = [(c, s, k) for c in range(len(configs)) for s in FOLD_SEEDS for k in range(N_FOLDS)]
    predicted = Parallel(n_jobs=-1, verbose=5)(
        delayed(fit_config)(folded[s, k], configs[c], fold_masks[s][k]) for c, s, k in fold_tasks)
    heldout = []
    for (c, s, k), (pred, _) in zip(fold_tasks, predicted):
        truth = labels[c, s][~fold_masks[s][k]]
        both = (pred != -1) & (truth != -1)
        heldout.append(dict(config=c, heldout_outliers=float(np.mean(pred == -1)),
                            heldout_ami=adjusted_mutual_info_score(truth[both], pred[both]) if both.sum() > 1
                            else np.nan))
    heldout = pd.DataFrame(heldout).groupby('config').mean()

    stable = pd.DataFrame([{'config': c, **stability([labels[c, s] for s in SEEDS])} for c in range(len(configs))])
    runs.to_csv(OUT / 'search_runs.csv', index=False)

    metrics = [m for m in runs.columns if m not in ['config', 'seed', *HDBSCAN_GRID]]
    summary = runs.groupby('config')[metrics].agg(['mean', 'std'])
    summary.columns = [f'{m}_sd' if stat == 'std' else m for m, stat in summary.columns]
    summary = (pd.DataFrame(configs).rename_axis('config').join(summary).join(stable.set_index('config'))
               .join(heldout).reset_index())
    summary.to_csv(OUT / 'search.csv', index=False)
    print(summary[['min_cluster_size', 'min_samples', 'cluster_selection_epsilon', 'n_topics', 'noise_frac',
                   'dbcv_score', 'npmi', 'diversity', 'keyword_score', 'stability_ami', 'stable_topics',
                   'heldout_outliers', 'heldout_ami']].round(3).to_string(index=False))


# ---------------------------------------------------------------------------------------------
# Stage report
# ---------------------------------------------------------------------------------------------

def pareto(table):
    """Mask of the configurations no other configuration beats on every objective."""
    values = np.column_stack([table[m] if up else -table[m] for m, up in OBJECTIVES.items()])
    front = np.ones(len(values), dtype=bool)
    for i, v in enumerate(values):
        if np.isnan(v).any():
            front[i] = False
            continue
        dominated = (values >= v).all(axis=1) & (values > v).any(axis=1)
        front[i] = not dominated.any()
    return front


def stage_report():
    table = pd.read_csv(OUT / 'search.csv')
    table['pareto'] = pareto(table)
    columns = ['config', *HDBSCAN_GRID, 'n_topics', 'noise_frac', 'size_min', 'size_median', *OBJECTIVES,
               'stability_jaccard', 'stable_topics', 'stability_kappa', 'heldout_ami', 'npmi', 'diversity']
    front = table.loc[table['pareto'], columns].sort_values('n_topics')
    front.to_csv(OUT / 'pareto.csv', index=False)

    panels = ['stability_ami', 'stable_topics', 'dbcv_score', 'keyword_score', 'heldout_outliers', 'noise_frac',
              'npmi', 'diversity', 'heldout_ami']
    fig = make_subplots(rows=3, cols=3, subplot_titles=panels)
    hover = [f"mcs={r.min_cluster_size} ms={r.min_samples} eps={r.cluster_selection_epsilon}"
             for r in table.itertuples()]
    for i, metric in enumerate(panels):
        sd = table.get(f'{metric}_sd')
        for on_front, colour in [(False, '#B0B0B0'), (True, '#D55E00')]:
            rows = table['pareto'] == on_front
            fig.add_trace(go.Scatter(
                x=table.loc[rows, 'n_topics'], y=table.loc[rows, metric], mode='markers',
                marker=dict(color=colour, size=8), name='Pareto front' if on_front else 'other',
                legendgroup=str(on_front), showlegend=i == 0, text=[hover[k] for k in np.flatnonzero(rows)],
                error_y=dict(array=sd[rows], thickness=1) if sd is not None else None),
                row=i // 3 + 1, col=i % 3 + 1)
    fig.update_xaxes(type='log', title_text='topics (mean over seeds)')
    fig.update_layout(height=1000, title='HDBSCAN configurations: metrics against the number of topics')
    fig.write_html(OUT / 'report.html', include_plotlyjs='cdn')
    print(f"{len(front)} of {len(table)} configurations on the Pareto front "
          f"({', '.join(f'{m} {chr(8593) if up else chr(8595)}' for m, up in OBJECTIVES.items())}):")
    print(front.round(3).to_string(index=False))
    print(f"\nChoose one, set config.HDBSCAN_PARAMS and record it and why in {OUT / 'CHOICE.md'}")


# ---------------------------------------------------------------------------------------------
# Stage consensus
# ---------------------------------------------------------------------------------------------

def consensus_metrics(c, linkage_tree, keep, threshold, min_size, neighbours, halves):
    """Metrics of the consensus `c` cut at `threshold`: topics, outliers, confidence, split-half agreement of the
    consensus of each half of the runs (`halves`: two (co-association, tree, keep)), and the leave-one-out vote with
    each number of neighbours of CONSENSUS_KNNS (`neighbours`: each paper's nearest other papers, nearest first)."""
    labels = consensus.cut(c, linkage_tree, keep, threshold, min_size)
    sizes = np.bincount(labels[labels != -1]) if (labels != -1).any() else np.array([0])
    split = consensus.agreement(*[consensus.cut(hc, ht, hk, threshold, min_size) for hc, ht, hk in halves])
    row = dict(threshold=threshold, n_topics=int((sizes > 0).sum()), noise_frac=float(np.mean(labels == -1)),
               size_min=int(sizes.min()), size_median=float(np.median(sizes)),
               confidence=float(np.nanmean(consensus.confidence(c, labels))) if (labels != -1).any() else np.nan,
               **{f'split_{k}': v for k, v in split.items()})
    clustered = labels != -1
    for k in CONSENSUS_KNNS:
        loo = consensus.vote(labels[neighbours[:, :k]])
        row[f'loo_kept_k{k}'] = float(np.mean(loo[clustered] == labels[clustered])) if clustered.any() else np.nan
        row[f'loo_unmatched_k{k}'] = float(np.mean(loo == -1))
    return row


def consensus_of(label_runs, threshold):
    """(co-association, tree, keep) of `label_runs`, with the tree for thresholds of at least `threshold`."""
    c = consensus.coassociation(label_runs)
    return (c, *consensus.tree(c, threshold))


def stage_consensus(embeddings, key, config_ids=None, converge=False):
    search = pd.read_csv(OUT / 'search.csv')
    # The Pareto front's results go to consensus.*; those of chosen configurations to consensus_selected.*
    name = 'consensus' if config_ids is None else 'consensus_selected'
    if config_ids is None:
        config_ids = pd.read_csv(OUT / 'pareto.csv')['config'].tolist()
    candidates = search.set_index('config').loc[config_ids, list(HDBSCAN_GRID)]
    nn, nc = config.UMAP_PARAMS['n_neighbors'], config.UMAP_PARAMS['n_components']
    n_runs = config.CONSENSUS_RUNS
    seeds = list(range(max(n_runs, 2 * max(CONVERGENCE_HALVES) if converge else 0)))
    print(f"{len(candidates)} configurations; consensus of {n_runs} runs (UMAP n_neighbors={nn}, n_components={nc}, "
          f"{len(seeds)} seeds)")
    reduced = Parallel(n_jobs=-1, verbose=5)(delayed(umap_embedding)(embeddings, key, nn, nc, s) for s in seeds)
    index = NearestNeighbors(n_neighbors=max(CONSENSUS_KNNS), metric='cosine', algorithm='brute').fit(embeddings)
    neighbours = index.kneighbors(return_distance=False)  # each paper's nearest other papers
    low = min(CONSENSUS_THRESHOLDS)

    rows, convergence = [], []
    for config_id, grid_values in candidates.iterrows():
        params = {**config.HDBSCAN_PARAMS, **grid_values.to_dict()}
        params['min_samples'] = int(params['min_samples'])
        params['min_cluster_size'] = int(params['min_cluster_size'])
        print(f"config {config_id}: {grid_values.to_dict()}")
        label_runs = Parallel(n_jobs=-1)(delayed(consensus.cluster)(r, params) for r in reduced)
        main = label_runs[:n_runs]
        full = consensus_of(main, low)
        halves = [consensus_of(main[:n_runs // 2], low), consensus_of(main[n_runs // 2:], low)]
        for threshold in CONSENSUS_THRESHOLDS:
            rows.append(dict(config=config_id, **grid_values.to_dict(),
                             **consensus_metrics(*full, threshold, params['min_cluster_size'], neighbours, halves)))
        if converge:
            for h in CONVERGENCE_HALVES:
                parts = [consensus_of(label_runs[:h], config.CONSENSUS_THRESHOLD),
                         consensus_of(label_runs[h:2 * h], config.CONSENSUS_THRESHOLD)]
                pair = consensus.agreement(*[consensus.cut(c, t, k, config.CONSENSUS_THRESHOLD,
                                                           params['min_cluster_size']) for c, t, k in parts])
                convergence.append(dict(config=config_id, runs_per_half=h, **pair))

    table = pd.DataFrame(rows)
    table.to_csv(OUT / f'{name}.csv', index=False)
    k = config.CONSENSUS_KNN if config.CONSENSUS_KNN in CONSENSUS_KNNS else CONSENSUS_KNNS[0]
    panels = ['n_topics', 'noise_frac', 'confidence', 'split_ami', 'split_kappa', 'split_jaccard',
              f'loo_kept_k{k}', f'loo_unmatched_k{k}']
    fig = make_subplots(rows=2, cols=4, subplot_titles=panels)
    for i, metric in enumerate(panels):
        for config_id, group in table.groupby('config'):
            fig.add_trace(go.Scatter(x=group['threshold'], y=group[metric], mode='lines+markers', name=str(config_id),
                                     legendgroup=str(config_id), showlegend=i == 0), row=i // 4 + 1, col=i % 4 + 1)
    fig.update_xaxes(title_text='threshold')
    fig.update_layout(height=750, title=f'Consensus of {n_runs} runs against the threshold, per configuration '
                                        f'(leave-one-out vote with {k} neighbours)', legend_title='config')
    fig.write_html(OUT / f'{name}.html', include_plotlyjs='cdn')
    columns = ['config', *HDBSCAN_GRID, 'threshold', 'n_topics', 'noise_frac', 'confidence', 'split_ami',
               'split_kappa', f'loo_kept_k{k}', f'loo_unmatched_k{k}']
    print(table[columns].round(3).to_string(index=False))
    if converge:
        convergence = pd.DataFrame(convergence)
        convergence.to_csv(OUT / 'convergence.csv', index=False)
        print("\nSplit-half agreement against the runs per half (threshold "
              f"{config.CONSENSUS_THRESHOLD}):")
        print(convergence.round(3).to_string(index=False))
    print(f"\nSet config.HDBSCAN_PARAMS and CONSENSUS_*, and record the choice and why in {OUT / 'CHOICE.md'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('stage', choices=['sizes', 'umap', 'search', 'report', 'consensus'])
    parser.add_argument('--sample', type=int, help="run on a random subset of this many papers")
    parser.add_argument('--quick', action='store_true', help="reduced grids, for checks")
    parser.add_argument('--configs', type=int, nargs='+',
                        help="consensus: the configurations (search.csv `config`) to evaluate; default the Pareto front")
    parser.add_argument('--converge', action='store_true',
                        help="consensus: also the split-half agreement against the number of runs")
    args = parser.parse_args()
    if args.quick:
        quick()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == 'report':
        return stage_report()
    data, embeddings, key = load(args.sample)
    if args.stage == 'sizes':
        stage_sizes(data)
    elif args.stage == 'umap':
        stage_umap(embeddings, key)
    elif args.stage == 'search':
        stage_search(data, embeddings, key)
    else:
        stage_consensus(embeddings, key, args.configs, args.converge)


if __name__ == '__main__':
    main()
