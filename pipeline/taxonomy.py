"""The versioned topic taxonomy of the live review.

Each major version (v1, v2, ...) is one fit of the topic model (baseline.py). Minor versions (v1.1, v1.2, ...) add
topics found among the unmatched papers (discover.py). Files:

  bertopic/taxonomy/versions.csv          one row per version: version, date, kind (baseline|addition), ...
  bertopic/taxonomy/runs.csv              one row per weekly assign.py run, with the share of unmatched papers
  bertopic/taxonomy/v<N>/topics.csv       topic, label, label_source, added_in, added_on, source_model, source_cluster
  bertopic/taxonomy/v<N>/assignments.csv  paper_id, topic, first_topic, method (fit|predict), version,
                                          assigned_on, moved_in
  bertopic/taxonomy/v<N>/hierarchy.csv    topic, level, parent, parent_label: the (fine) topics merged up to each
                                          level of config.TOPIC_LEVELS
  bertopic/taxonomy/v<N>/merges.csv       level, small, small_size, nearest, nearest_size, similarity, parent: each
                                          merge of build_hierarchy, in order
  bertopic/taxonomy/v<N>/labels.csv       level, topic, label, description, keywords, label_source, model,
                                          labelled_on: the AI-written name of each topic at each level (labels.py),
                                          used instead of the keyword labels of topics.csv and hierarchy.csv
  bertopic/models/v<N>/base.pkl           the fitted BERTopic of the baseline (a consensus model, see consensus.py)
  bertopic/models/v<N>/params.json        its parameters, agreement, expected unmatched share and environment
  bertopic/models/v<N>/addition_<k>.pkl   the model fitted on the unmatched papers of addition k

`method`, `version` and `first_topic` record how and when a paper was first assigned; `moved_in` is the version in
which discover.py moved it into an added topic (empty otherwise). A topic is produced by `source_model` (base or
addition_<k>) as its topic `source_cluster`. Older major versions are never modified. Topics added after the
baseline are their own parent at every level until the next baseline.
"""
import copy
import datetime as dt
import json

import joblib
import numpy as np
import pandas as pd
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from sklearn.base import clone

import config

VERSIONS = config.TAXONOMY_DIR / 'versions.csv'
RUNS = config.TAXONOMY_DIR / 'runs.csv'
OUTLIER_LABEL = '-1: Outliers'


def today():
    return dt.date.today().isoformat()


def versions():
    return pd.read_csv(VERSIONS) if VERSIONS.exists() else pd.DataFrame(columns=['version'])


def current_version():
    """Latest version, e.g. 'v1.2'. Raises if there is no baseline yet."""
    v = versions()
    if not len(v):
        raise SystemExit("No taxonomy yet: run pipeline/baseline.py first")
    return v['version'].iloc[-1]


def major(version):
    return version.split('.')[0]


def add_version(version, kind, model, n_topics, n_papers, note=''):
    row = pd.DataFrame([dict(version=version, date=today(), kind=kind, model=model, n_topics=n_topics,
                             n_papers=n_papers, note=note)])
    row.to_csv(VERSIONS, mode='a', header=not VERSIONS.exists(), index=False)


def _dir(major_version):
    return config.TAXONOMY_DIR / major_version


def load_topics(major_version=None):
    return pd.read_csv(_dir(major_version or major(current_version())) / 'topics.csv')


def load_assignments(major_version=None):
    return pd.read_csv(_dir(major_version or major(current_version())) / 'assignments.csv')


def save_topics(topics, major_version):
    _dir(major_version).mkdir(parents=True, exist_ok=True)
    topics.to_csv(_dir(major_version) / 'topics.csv', index=False)


def save_assignments(assignments, major_version):
    _dir(major_version).mkdir(parents=True, exist_ok=True)
    assignments.to_csv(_dir(major_version) / 'assignments.csv', index=False)


def labels(major_version=None, level=None):
    """Full label of each topic, e.g. {3: '3: Plastic waste management and policy'}, outliers included.

    With `level`, the labels of the topics of that level (topics added after the baseline keep theirs). The AI labels
    of labels.csv are used where they exist, otherwise the keyword labels of topics.csv and hierarchy.csv."""
    topics = load_topics(major_version)
    result = {-1: OUTLIER_LABEL, **{t: f"{t}: {label}" for t, label in zip(topics['topic'], topics['label'])}}
    if level is not None:
        rows = load_hierarchy(major_version).query('level == @level')
        merged = rows[rows['parent'] != rows['topic']]
        result.update({p: f"{p}: {label}" for p, label in zip(merged['parent'], merged['parent_label'])})
    ai = load_labels(major_version)
    ai = ai[ai['level'] == (level or 'fine')]
    result.update({t: f"{t}: {label}" for t, label in zip(ai['topic'], ai['label'])})
    return result


def labels_path(major_version=None):
    return _dir(major_version or major(current_version())) / 'labels.csv'


LABEL_COLUMNS = ['level', 'topic', 'label', 'description', 'keywords', 'label_source', 'model', 'labelled_on']


def load_labels(major_version=None):
    """labels.csv of a major version: the AI-written label and description of each topic at each level (empty if
    labels.py has not run)."""
    path = labels_path(major_version)
    return pd.read_csv(path) if path.exists() else pd.DataFrame(columns=LABEL_COLUMNS)


def build_hierarchy(topics, embeddings, levels):
    """Merge the topics up to each level of `levels` ({name: minimum papers per topic}).

    The smallest topic is merged into its most similar topic (cosine of the centroids of the mean-centred
    embeddings) until every topic of the level has at least the level's minimum number of papers. Levels are taken
    in increasing order of their minimum and continue the same merges, so they are nested. A merged topic keeps the
    id of the larger of the two. Returns (one row per (topic, level) with its `parent` at that level, one row per
    merge with the similarity it was made at: a low one means a small topic had no close neighbour)."""
    topics = np.asarray(topics)
    centred = embeddings - embeddings.mean(axis=0)
    members = {t: {t} for t in np.unique(topics[topics != -1])}
    sums = {t: centred[topics == t].sum(axis=0) for t in members}
    sizes = {t: int(np.sum(topics == t)) for t in members}
    rows, merges = [], []
    for level, minimum in sorted(levels.items(), key=lambda item: item[1]):
        while len(members) > 1 and min(sizes.values()) < minimum:
            small = min(sizes, key=lambda t: (sizes[t], t))
            others = [t for t in members if t != small]
            vectors = np.array([sums[t] for t in others])
            similarity = vectors @ sums[small] / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(sums[small]))
            other = others[int(np.argmax(similarity))]
            keep, drop = (other, small) if sizes[other] >= sizes[small] else (small, other)
            merges.append(dict(level=level, small=small, small_size=sizes[small], nearest=other,
                               nearest_size=sizes[other], similarity=round(float(similarity.max()), 4), parent=keep))
            members[keep] |= members.pop(drop)
            sums[keep] = sums[keep] + sums.pop(drop)
            sizes[keep] += sizes.pop(drop)
        parent = {t: p for p, fine in members.items() for t in fine}
        rows += [dict(topic=t, level=level, parent=parent[t]) for t in sorted(parent)]
    return (pd.DataFrame(rows, columns=['topic', 'level', 'parent']),
            pd.DataFrame(merges, columns=['level', 'small', 'small_size', 'nearest', 'nearest_size', 'similarity',
                                          'parent']))


def save_hierarchy(hierarchy, merges, major_version):
    _dir(major_version).mkdir(parents=True, exist_ok=True)
    hierarchy.to_csv(_dir(major_version) / 'hierarchy.csv', index=False)
    merges.to_csv(_dir(major_version) / 'merges.csv', index=False)


def load_hierarchy(major_version=None):
    major_version = major_version or major(current_version())
    path = _dir(major_version) / 'hierarchy.csv'
    if not path.exists():
        raise SystemExit(f"{major_version} has no hierarchy.csv (it predates the hierarchy); the topic levels of "
                         "config.TOPIC_LEVELS need a baseline made by the current baseline.py")
    return pd.read_csv(path)


def level_topics(topics, level, major_version=None):
    """The topic of each paper at `level` (None for the fine topics): its fine topic's parent at that level."""
    topics = np.asarray(topics)
    if level is None:
        return topics
    rows = load_hierarchy(major_version).query('level == @level')
    if not len(rows):
        raise KeyError(f"No level {level!r} in hierarchy.csv; the levels are those of config.TOPIC_LEVELS")
    parent = dict(zip(rows['topic'], rows['parent']))
    return np.array([parent.get(t, t) for t in topics])


def load_params(major_version=None):
    """params.json of a major version's base model ({} if it has none, as v1)."""
    path = config.MODELS_DIR / (major_version or major(current_version())) / 'params.json'
    return json.loads(path.read_text()) if path.exists() else {}


def model_path(major_version, name='base'):
    return config.MODELS_DIR / major_version / f'{name}.pkl'


def save_model(topic_model, major_version, name='base'):
    """Pickle the model for assign.py, without the embedding model and the fitted c-TF-IDF (refitted by analysis.py
    over the current papers anyway), which keeps the file small enough for git. Labels must be taken before."""
    path = model_path(major_version, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    topic_model = copy.copy(topic_model)
    topic_model.embedding_model = None
    topic_model.vectorizer_model = clone(topic_model.vectorizer_model)
    topic_model.ctfidf_model = ClassTfidfTransformer()
    topic_model.c_tf_idf_ = None
    joblib.dump(topic_model, path, compress=3)


def load_model(major_version, name='base', embedding_model=None):
    return BERTopic.load(str(model_path(major_version, name)), embedding_model=embedding_model)


def keyword_label(topic_model, topic, n_words=4):
    """Provisional label of a topic: its top c-TF-IDF words."""
    return ', '.join(word for word, _ in topic_model.get_topic(topic)[:n_words])
