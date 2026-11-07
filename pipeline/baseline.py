"""Fit a new topic model on all papers and open a new major version of the taxonomy (v1, v2, ...).

Run it for the first live model and whenever assign.py flags a re-baseline. The topics are the consensus of
config.CONSENSUS_RUNS UMAP + HDBSCAN fits with different seeds (see consensus.py), with the parameters of config.py
(chosen with evaluate.py). It writes:
  bertopic/models/v<N>/base.pkl, params.json         the fitted model and its parameters, with the agreement of the
                                                     consensus of the first and second half of the seeds, the
                                                     leave-one-out vote of the training papers and the environment
  bertopic/models/v<N>/consensus.csv                 each paper's mean co-association with the papers of its topic
  bertopic/taxonomy/v<N>/topics.csv, assignments.csv  provisional keyword labels and every paper's topic
  bertopic/taxonomy/v<N>/hierarchy.csv, merges.csv   the topics merged up to the levels of config.TOPIC_LEVELS, and
                                                     each merge with its similarity
  bertopic/taxonomy/v<N>/crosswalk.csv               each previous topic matched to a new one (Hungarian matching on
                                                     shared papers), and crosswalk_counts.csv (papers per pair)
The previous version is the last major version (none for the first). Older versions are left as they are.
"""
import argparse
import copy
import json
import platform
import subprocess
from importlib import metadata

import numpy as np
import pandas as pd
from bertopic import BERTopic
from bertopic.dimensionality import BaseDimensionalityReduction
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import adjusted_rand_score

import config
import consensus
import taxonomy

# Crosswalk: Hungarian pairs with a lower Jaccard index are not matches. A topic is split (merged) when at least
# SPLIT_SHARE of its papers go to (come from) each of several topics of the other version.
MIN_JACCARD = 0.1
SPLIT_SHARE = 0.2
PACKAGES = ['bertopic', 'umap-learn', 'hdbscan', 'scikit-learn', 'scipy', 'numpy', 'pandas', 'sentence-transformers',
            'torch', 'transformers']


def fit(docs, embeddings, paper_ids, umap_params, hdbscan_params, n_runs=None):
    """Consensus topic model of the papers `paper_ids`: (topic model, topics, info).

    `info` has the number of runs, the agreement (consensus.agreement) between the consensus of the first and second
    half of the seeds (reproducibility of the partition), each paper's confidence (see consensus.confidence), and the
    leave-one-out vote of the training papers: the share that keep their topic, and the share left unmatched, which
    is what to expect of new papers like them."""
    n_runs = n_runs or config.CONSENSUS_RUNS
    label_runs = consensus.runs(embeddings, umap_params, hdbscan_params, range(n_runs))

    def agree(selected):
        return consensus.partition(consensus.coassociation(selected), config.CONSENSUS_THRESHOLD,
                                   hdbscan_params['min_cluster_size'])

    labels, confidence = agree(label_runs)
    half = n_runs // 2
    split = (consensus.agreement(agree(label_runs[:half])[0], agree(label_runs[half:])[0]) if half >= 2
             else dict(ari=np.nan, ami=np.nan, kappa=np.nan, jaccard=np.nan))
    print(f"Consensus of {n_runs} runs: {labels.max() + 1} topics, {np.mean(labels == -1):.1%} outliers; agreement "
          f"of the two halves of the seeds: " + ', '.join(f"{k} {v:.2f}" for k, v in split.items()))

    cluster_model = consensus.ConsensusCluster(paper_ids, config.CONSENSUS_KNN)
    topic_model = BERTopic(umap_model=BaseDimensionalityReduction(), hdbscan_model=cluster_model,
                           # Exclude English stop words; bigrams keep compound words like "plastic waste"
                           vectorizer_model=CountVectorizer(ngram_range=(1, 2), stop_words="english"),
                           verbose=True)
    topics, _ = topic_model.fit_transform(docs, embeddings, y=labels)
    loo = cluster_model.loo_predict()  # in the consensus' own topic ids, like `labels`
    clustered = labels != -1
    vote = dict(kept=float(np.mean(loo[clustered] == labels[clustered])) if clustered.any() else np.nan,
                unmatched=float(np.mean(loo == -1)))
    print(f"Leave-one-out vote: {vote['kept']:.1%} of the clustered papers keep their topic; "
          f"{vote['unmatched']:.1%} of all papers unmatched")
    return topic_model, np.asarray(topics), dict(runs=n_runs, split=split, confidence=confidence, vote=vote)


def level_labels(topic_model, docs, topics, hierarchy):
    """Keyword label of each merged topic of `hierarchy`, from the c-TF-IDF of its papers."""
    labels = {}
    for level, rows in hierarchy.groupby('level', sort=False):
        if (rows['parent'] == rows['topic']).all():  # no merges: the topics' own labels
            labels.update({(level, p): taxonomy.keyword_label(topic_model, p) for p in rows['parent']})
            continue
        mapping = dict(zip(rows['topic'], rows['parent']))
        merged = copy.deepcopy(topic_model)
        merged.update_topics(docs, topics=[mapping.get(t, t) for t in topics], vectorizer_model=merged.vectorizer_model)
        labels.update({(level, p): taxonomy.keyword_label(merged, p) for p in rows['parent'].unique()})
    return [labels[level, parent] for level, parent in zip(hierarchy['level'], hierarchy['parent'])]


def centroids(embeddings, topics):
    """Mean embedding of each topic (outliers excluded), as {topic: unit vector}."""
    result = {}
    for topic in np.unique(topics[topics != -1]):
        c = embeddings[topics == topic].mean(axis=0)
        result[topic] = c / np.linalg.norm(c)
    return result


def crosswalk(old, new, embeddings, old_labels, new_labels):
    """Match each old topic to one new topic by the overlap of their papers (Jaccard index, Hungarian matching).

    Pairs with a Jaccard index below MIN_JACCARD are not matches: both topics are listed as without a counterpart.
    `share_of_old` / `share_of_new` are the shared papers as a share of each topic; `old_split_into` is the number of
    new topics that each have at least SPLIT_SHARE of the old topic's papers, `new_merged_from` the number of old
    topics that each make up at least SPLIT_SHARE of the new topic. The cosine similarity of the topic centroids (of
    the mean-centred embeddings, as in taxonomy.build_hierarchy) is reported too, but not matched on."""
    counts = pd.crosstab(pd.Series(old, name='old_topic'), pd.Series(new, name='new_topic'))
    old_ids = [t for t in counts.index if t != -1]
    new_ids = [t for t in counts.columns if t != -1]
    shared = counts.loc[old_ids, new_ids].to_numpy()
    old_size, new_size = pd.Series(old).value_counts(), pd.Series(new).value_counts()
    old_n, new_n = old_size[old_ids].to_numpy(), new_size[new_ids].to_numpy()
    jaccard = shared / (old_n[:, None] + new_n[None, :] - shared)
    rows, cols = linear_sum_assignment(-jaccard)
    matched = jaccard[rows, cols] >= MIN_JACCARD
    rows, cols = rows[matched], cols[matched]
    centred = embeddings - embeddings.mean(axis=0)
    old_c, new_c = centroids(centred, old), centroids(centred, new)

    table = pd.DataFrame({'old_topic': [old_ids[r] for r in rows], 'new_topic': [new_ids[c] for c in cols],
                          'jaccard': jaccard[rows, cols].round(3), 'shared_papers': shared[rows, cols],
                          'share_of_old': (shared[rows, cols] / old_n[rows]).round(3),
                          'share_of_new': (shared[rows, cols] / new_n[cols]).round(3)})
    table['centroid_similarity'] = [round(float(old_c[o] @ new_c[n]), 3)
                                    for o, n in zip(table['old_topic'], table['new_topic'])]
    # Topics without a counterpart (different numbers of topics, or no match above MIN_JACCARD) are listed too
    table = pd.concat([table,
                       pd.DataFrame({'old_topic': [o for o in old_ids if o not in set(table['old_topic'])]}),
                       pd.DataFrame({'new_topic': [n for n in new_ids if n not in set(table['new_topic'])]})],
                      ignore_index=True)
    table['old_size'] = table['old_topic'].map(old_size)
    table['new_size'] = table['new_topic'].map(new_size)
    table['old_split_into'] = table['old_topic'].map(
        dict(zip(old_ids, (shared / old_n[:, None] >= SPLIT_SHARE).sum(axis=1))))
    table['new_merged_from'] = table['new_topic'].map(
        dict(zip(new_ids, (shared / new_n[None, :] >= SPLIT_SHARE).sum(axis=0))))
    integer = ['old_topic', 'new_topic', 'shared_papers', 'old_size', 'new_size', 'old_split_into', 'new_merged_from']
    table[integer] = table[integer].astype('Int64')
    table['old_label'] = table['old_topic'].map(old_labels)
    table['new_label'] = table['new_topic'].map(new_labels)
    return table.sort_values(['old_topic', 'new_topic'], na_position='last'), counts


def previous_assignments(ids):
    """(name, topics aligned with `ids` or -2 where unknown, labels) of the previous major version, or None."""
    v = taxonomy.versions()
    if not len(v):
        return None
    prev = taxonomy.major(v['version'].iloc[-1])
    a = taxonomy.load_assignments(prev).set_index('paper_id')['topic']
    return prev, ids.map(a).fillna(-2).astype(int).to_numpy(), taxonomy.labels(prev)


def environment():
    """Python, platform, package versions and git commit of the fit."""
    packages = {}
    for package in PACKAGES:
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            pass

    def git(*args):
        try:
            return subprocess.run(['git', '-C', str(config.ROOT), *args], capture_output=True, text=True,
                                  check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = git('status', '--porcelain')
    return dict(python=platform.python_version(), platform=platform.platform(), packages=packages,
                git_commit=git('rev-parse', 'HEAD'), git_dirty=bool(status) if status is not None else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()

    data = pd.read_csv(config.ALLDATA)
    ids = config.paper_ids(data)
    docs = data['Abstract'].tolist()
    embeddings = config.load_embeddings(ids)
    umap_params, hdbscan_params = dict(config.UMAP_PARAMS), dict(config.HDBSCAN_PARAMS)

    n_majors = taxonomy.versions()['version'].map(taxonomy.major).nunique()
    major = f"v{n_majors + 1}"
    version = f"{major}.0"
    previous = previous_assignments(ids)

    topic_model, topics, info = fit(docs, embeddings, ids, umap_params, hdbscan_params)
    topic_ids = sorted(t for t in set(topics) if t != -1)
    print(f"{len(topic_ids)} topics, {np.mean(topics == -1):.1%} outliers")

    topics_df = pd.DataFrame({'topic': topic_ids,
                              'label': [taxonomy.keyword_label(topic_model, t) for t in topic_ids],
                              'label_source': 'keywords', 'added_in': version, 'added_on': taxonomy.today(),
                              'source_model': 'base', 'source_cluster': topic_ids})
    taxonomy.save_topics(topics_df, major)
    assignments = pd.DataFrame({'paper_id': ids, 'topic': topics, 'first_topic': topics, 'method': 'fit',
                                'version': version, 'assigned_on': taxonomy.today(), 'moved_in': None})
    taxonomy.save_assignments(assignments, major)

    hierarchy, merges = taxonomy.build_hierarchy(topics, embeddings, config.TOPIC_LEVELS)
    hierarchy['parent_label'] = level_labels(topic_model, docs, topics, hierarchy)
    taxonomy.save_hierarchy(hierarchy, merges, major)
    print("Topics per level:", hierarchy.groupby('level', sort=False)['parent'].nunique().to_dict())

    taxonomy.save_model(topic_model, major)
    params = {'date': taxonomy.today(), 'n_papers': len(ids), 'encoder': config.encoder(),
              'umap': umap_params, 'hdbscan': hdbscan_params,
              'consensus': {'runs': info['runs'], 'threshold': config.CONSENSUS_THRESHOLD,
                            'knn': config.CONSENSUS_KNN,
                            'split_half': {k: round(v, 3) for k, v in info['split'].items()}},
              'fit': {'topics': len(topic_ids), 'outlier_share': round(float(np.mean(topics == -1)), 4),
                      'loo_kept': round(info['vote']['kept'], 4),
                      'expected_unmatched': round(info['vote']['unmatched'], 4)},
              'topic_levels': config.TOPIC_LEVELS,
              'choice': 'bertopic/evaluation/CHOICE.md',
              'environment': environment()}
    (config.MODELS_DIR / major / 'params.json').write_text(json.dumps(params, indent=2) + '\n')
    pd.DataFrame({'paper_id': ids, 'confidence': np.round(info['confidence'], 3)}).to_csv(
        config.MODELS_DIR / major / 'consensus.csv', index=False)

    note = (f"consensus of {info['runs']} runs, split-half ARI {info['split']['ari']:.2f} "
            f"AMI {info['split']['ami']:.2f}")
    # Crosswalk from the previous version, on the papers it covers
    if previous is not None:
        prev_name, prev_topics, prev_labels = previous
        known = prev_topics != -2
        if known.any():
            table, counts = crosswalk(prev_topics[known], topics[known], embeddings[known], prev_labels,
                                      taxonomy.labels(major))
            table.to_csv(config.TAXONOMY_DIR / major / 'crosswalk.csv', index=False)
            counts.to_csv(config.TAXONOMY_DIR / major / 'crosswalk_counts.csv')
            ari = adjusted_rand_score(prev_topics[known], topics[known])
            note += f"; ARI {ari:.2f} with {prev_name}"
            print(f"Adjusted Rand index with {prev_name}: {ari:.2f}")

    taxonomy.add_version(version, 'baseline', f"models/{major}/base.pkl", len(topic_ids), len(ids), note=note)
    print(f"Taxonomy {version} written. Run analysis.py to update the figures.")


if __name__ == '__main__':
    main()
