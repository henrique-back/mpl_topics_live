"""Weekly: give every paper of alldata.csv without a topic one from the current taxonomy, without changing the others.

Each new paper is placed by the base model's transform(): from v2, the vote of its nearest fitted papers in the
embedding space (consensus.ConsensusCluster); v1 used its fitted UMAP and HDBSCAN (approximate_predict). Papers it
leaves unmatched (-1) are then passed through each addition's model in order (topics added by discover.py), and take
the topic of the first accepted cluster they fall in. The rest stay at -1.

Appends a row to bertopic/taxonomy/runs.csv with the share of unmatched papers among those predicted since the
baseline, and flags a re-baseline (baseline.py) when it exceeds the share expected of papers like the fitted ones (the
leave-one-out vote of params.json, `expected_unmatched`) by more than config.REBASELINE_POOL_SHARE.
"""
import numpy as np
import pandas as pd

import config
import taxonomy


def predict(docs, embeddings, major):
    """Topic of each paper: base model first, then each addition for the papers still unmatched."""
    topics = np.asarray(taxonomy.load_model(major).transform(docs, embeddings)[0])
    added = taxonomy.load_topics(major).query("source_model != 'base'")
    for source, group in added.groupby('source_model', sort=False):
        todo = np.flatnonzero(topics == -1)
        if not len(todo):
            break
        clusters = taxonomy.load_model(major, source).transform([docs[i] for i in todo], embeddings[todo])[0]
        # Clusters of the addition's model that were not accepted as topics stay unmatched
        topics[todo] = pd.Series(clusters).map(dict(zip(group['source_cluster'], group['topic']))).fillna(-1)
    return topics


def main():
    version = taxonomy.current_version()
    major = taxonomy.major(version)
    data = pd.read_csv(config.ALLDATA)
    data['paper_id'] = config.paper_ids(data)
    assignments = taxonomy.load_assignments(major)
    new = data[~data['paper_id'].isin(assignments['paper_id'])]
    print(f"Assigning {len(new)} new papers (taxonomy {version})")

    if not len(new):
        return

    topics = predict(new['Abstract'].tolist(), config.load_embeddings(new['paper_id']), major)
    rows = pd.DataFrame({'paper_id': new['paper_id'].values, 'topic': topics, 'first_topic': topics,
                         'method': 'predict', 'version': version, 'assigned_on': taxonomy.today(), 'moved_in': None})
    assignments = pd.concat([assignments, rows], ignore_index=True)
    taxonomy.save_assignments(assignments, major)
    n_unmatched = int(np.sum(topics == -1))
    print(f"{n_unmatched} of them unmatched")

    # Share of unmatched papers among those placed by prediction since the baseline (papers that an addition has
    # since moved into a new topic count as matched)
    predicted = assignments[assignments['method'] == 'predict']
    pool_share = float((predicted['topic'] == -1).mean()) if len(predicted) else 0.0
    expected = taxonomy.load_params(major).get('fit', {}).get('expected_unmatched', 0.0)
    due = config.REBASELINE_POOL_SHARE is not None and pool_share - expected > config.REBASELINE_POOL_SHARE
    run = pd.DataFrame([dict(date=taxonomy.today(), version=version, n_new=len(new), n_unmatched=n_unmatched,
                             n_predicted=len(predicted), pool_share=round(pool_share, 4),
                             expected_unmatched=expected, rebaseline_due=due)])
    run.to_csv(taxonomy.RUNS, mode='a', header=not taxonomy.RUNS.exists(), index=False)
    print(f"Unmatched share since the baseline: {pool_share:.1%} of {len(predicted)} papers "
          f"(expected {expected:.1%})")
    if due:
        print(f"\n*** RE-BASELINE DUE: the unmatched share exceeds the expected one by more than "
              f"{config.REBASELINE_POOL_SHARE:.1%}. Run pipeline/baseline.py. ***\n")


if __name__ == '__main__':
    main()
