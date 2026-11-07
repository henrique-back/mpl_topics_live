"""Periodically (monthly or quarterly): add the topics that have formed among the unmatched papers.

A topic model with the parameters of config.py is fitted on the pool of unmatched papers (pool_ids). Each of its
clusters whose centroid has a cosine similarity below config.NOVELTY_MAX_SIMILARITY to the centroid of every current
topic becomes a new topic: the novelty test of BERTopic.merge_models. HDBSCAN's min_cluster_size already holds a new
topic to the same size bar as the baseline's topics. The fitted model is saved as bertopic/models/v<N>/addition_<k>.pkl,
so that assign.py can place future papers in the new topics, and the pool papers in accepted clusters move into them,
whatever their age. The result is a new minor version (e.g. v1.1 -> v1.2) with provisional keyword labels.

Not usable yet: the pool and the similarity threshold are defined in the session on unmatched papers.
"""
import numpy as np
import pandas as pd

import baseline
import config
import taxonomy


def pool_ids(assignments, data):
    """paper_ids of the unmatched papers to search for new topics in."""
    raise NotImplementedError("The pool of unmatched papers is defined in the session on unmatched papers")


def main():
    if config.NOVELTY_MAX_SIMILARITY is None:
        raise SystemExit("config.NOVELTY_MAX_SIMILARITY is not set yet")
    version = taxonomy.current_version()
    major = taxonomy.major(version)
    assignments = taxonomy.load_assignments(major)
    topics = taxonomy.load_topics(major)
    data = pd.read_csv(config.ALLDATA)
    data['paper_id'] = config.paper_ids(data)

    pool = data[data['paper_id'].isin(pool_ids(assignments, data))]
    print(f"Searching {len(pool)} unmatched papers for new topics (taxonomy {version})")
    if len(pool) < config.HDBSCAN_PARAMS['min_cluster_size']:
        return
    pool_embeddings = config.load_embeddings(pool['paper_id'])
    candidate, clusters, _ = baseline.fit(pool['Abstract'].tolist(), pool_embeddings, pool['paper_id'],
                                          config.UMAP_PARAMS, config.HDBSCAN_PARAMS)

    # Centroids of the current topics, from all their papers
    topic_of = assignments.set_index('paper_id')['topic']
    current = baseline.centroids(config.load_embeddings(topic_of.index), topic_of.to_numpy())
    current = np.array(list(current.values()))
    accepted = []
    for cluster, centroid in baseline.centroids(pool_embeddings, clusters).items():
        similarity = float((current @ centroid).max())
        print(f"Cluster {cluster} ({np.sum(clusters == cluster)} papers): max similarity {similarity:.3f}")
        if similarity < config.NOVELTY_MAX_SIMILARITY:
            accepted.append(cluster)
    if not accepted:
        print("No new topic")
        return

    new_version = f"{major}.{int(version.split('.')[1]) + 1}"
    k = topics.loc[topics['source_model'] != 'base', 'source_model'].nunique() + 1
    taxonomy.save_model(candidate, major, f'addition_{k}')
    new_ids = dict(zip(accepted, range(topics['topic'].max() + 1, topics['topic'].max() + 1 + len(accepted))))
    new_topics = pd.DataFrame({'topic': list(new_ids.values()),
                               'label': [taxonomy.keyword_label(candidate, c) for c in accepted],
                               'label_source': 'keywords', 'added_in': new_version, 'added_on': taxonomy.today(),
                               'source_model': f'addition_{k}', 'source_cluster': accepted})
    taxonomy.save_topics(pd.concat([topics, new_topics], ignore_index=True), major)

    # Pool papers in accepted clusters move into the new topics
    moved = pd.Series(clusters, index=pool['paper_id'].values).map(new_ids).dropna().astype(int)
    rows = assignments['paper_id'].isin(moved.index)
    assignments.loc[rows, 'topic'] = assignments.loc[rows, 'paper_id'].map(moved)
    assignments.loc[rows, 'moved_in'] = new_version
    taxonomy.save_assignments(assignments, major)

    taxonomy.add_version(new_version, 'addition', f"models/{major}/addition_{k}.pkl", len(topics) + len(accepted),
                         len(assignments), note=f"{len(accepted)} topics added, {len(moved)} papers moved in")
    print(f"Taxonomy {new_version}: {len(accepted)} topics added, {len(moved)} papers moved into them")


if __name__ == '__main__':
    main()
