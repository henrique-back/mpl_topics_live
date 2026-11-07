"""AI-written label and description of every topic of the current taxonomy, at each level of the hierarchy.

Claude (ai.py) names the topics of one level at a time, fine, then mid, then coarse, so that the names within a level
are distinct and a broader topic is named knowing its subtopics. For each topic it sees its top c-TF-IDF keywords at
that level, its papers closest to the topic centroid (title and abstract) and, above the fine level, the names of its
subtopics. Written for a wider public: a short plain-language name and a one- or two-sentence description.

Writes bertopic/taxonomy/v<N>/labels.csv (see taxonomy.py), which taxonomy.labels() then uses everywhere. Only topics
without a label are named, so it does nothing until a baseline (baseline.py) or an addition (discover.py) brings new
topics. `--level` names only that level; `--redo` names every topic again.
"""
import argparse
import copy

import numpy as np
import pandas as pd

import ai
import config
import taxonomy

N_KEYWORDS = 10
N_PAPERS = 5
ABSTRACT_CHARS = 1200
LEVELS = ['fine', 'mid', 'coarse']  # bottom-up: each level is named knowing the names of the one below

SYSTEM = """You name the research topics of a living review of the scientific literature on microplastics and \
marine plastic pollution. The topics were found by clustering paper abstracts. Your names appear on a public website \
for a wide audience (the public, journalists, policymakers), so write plain English that a non-specialist \
understands: no abbreviations (write "microplastics", not "MPs"), no jargon where a common word exists, no hype.

For each topic, write:
- label: a short name of at most 6 words, in sentence case, naming what the research studies (e.g. "Microplastics \
in rivers and estuaries", "Plastic ingestion by seabirds"). Don't start with "Microplastics" unless needed, and \
don't use "Research on" or "Studies of".
- description: one or two sentences (at most 40 words) saying what the papers of this topic study and why it matters.

Names within a level must be distinct: when two topics are close, make the name say what sets each apart. Base the \
names on the papers and keywords given, not on assumptions."""

SCHEMA = {'type': 'object', 'additionalProperties': False, 'required': ['topics'],
          'properties': {'topics': {'type': 'array', 'items': {
              'type': 'object', 'additionalProperties': False, 'required': ['topic', 'label', 'description'],
              'properties': {'topic': {'type': 'integer'}, 'label': {'type': 'string'},
                             'description': {'type': 'string'}}}}}}


def keywords(topic_model, docs, topics):
    """Top c-TF-IDF words of each topic of `topics` (one per paper), refitted over these papers."""
    model = copy.deepcopy(topic_model)
    model.update_topics(docs, topics=list(topics), vectorizer_model=model.vectorizer_model)
    return {t: [word for word, _ in model.get_topic(t)[:N_KEYWORDS]] for t in set(topics) if t != -1}


def central_papers(embeddings, topics, topic, n=N_PAPERS):
    """Positions of the `n` papers of `topic` closest (cosine) to its centroid."""
    members = np.flatnonzero(topics == topic)
    vectors = embeddings[members]
    centroid = vectors.mean(axis=0)
    similarity = vectors @ centroid / np.linalg.norm(centroid)  # the embeddings are L2-normalized
    return members[np.argsort(-similarity)[:n]]


def describe(topic, size, words, papers, children, data):
    lines = [f"## Topic {topic} ({size} papers)", f"Keywords: {', '.join(words)}"]
    if children:
        lines.append(f"Subtopics: {'; '.join(children)}")
    lines.append("Papers closest to the topic centre:")
    for i in papers:
        abstract = data['Abstract'].iloc[i]
        abstract = abstract[:ABSTRACT_CHARS] + ('...' if len(abstract) > ABSTRACT_CHARS else '')
        lines.append(f"- {data['Title'].iloc[i]}: {abstract}")
    return '\n'.join(lines)


def name_level(level, to_name, named, prompt_topics):
    """Ask Claude for the labels of the topics `to_name` of `level`; `named` are the level's existing labels."""
    intro = (f"Name the {len(to_name)} topics below. They are the "
             f"{ {'fine': 'most detailed', 'mid': 'medium', 'coarse': 'broadest'}[level] } level of a three-level "
             "hierarchy of research topics.")
    if named:
        intro += (" Other topics of this level are already named; keep your names distinct from them: "
                  + '; '.join(named) + '.')
    reply = ai.ask(intro + '\n\n' + '\n\n'.join(prompt_topics), SCHEMA, SYSTEM)['topics']
    answered = {row['topic']: row for row in reply}
    missing = set(to_name) - set(answered)
    if missing:
        raise RuntimeError(f"Claude named no label for topics {sorted(missing)} of level {level}")
    return [answered[t] for t in to_name]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--level', choices=LEVELS, help='name only this level')
    parser.add_argument('--redo', action='store_true', help='name every topic again, replacing the existing labels')
    args = parser.parse_args()

    major = taxonomy.major(taxonomy.current_version())
    existing = taxonomy.load_labels(major)
    if args.redo:
        existing = existing[existing['level'] != args.level] if args.level else existing.iloc[:0]

    data = pd.read_csv(config.ALLDATA)
    data['paper_id'] = config.paper_ids(data)
    fine = data['paper_id'].map(taxonomy.load_assignments(major).set_index('paper_id')['topic'])
    if fine.isna().any():
        raise SystemExit(f"{fine.isna().sum()} papers have no topic; run assign.py")
    fine = fine.astype(int).to_numpy()
    at = {level: taxonomy.level_topics(fine, level, major) for level in LEVELS}

    todo = {level: sorted(set(at[level]) - {-1} - set(existing.loc[existing['level'] == level, 'topic']))
            for level in (args.level and [args.level] or LEVELS)}
    if not any(todo.values()):
        print(f"Every topic of {major} has a label")
        return

    embeddings = config.load_embeddings(data['paper_id'])
    topic_model = taxonomy.load_model(major)
    docs = data['Abstract'].tolist()
    labels = existing.copy()
    for level in LEVELS:
        if not todo.get(level):
            continue
        words = keywords(topic_model, docs, at[level])
        below = LEVELS[LEVELS.index(level) - 1] if level != 'fine' else None
        below_labels = dict(labels.loc[labels['level'] == below, ['topic', 'label']].values) if below else {}
        prompt_topics = []
        for t in todo[level]:
            children = sorted(set(at[below][at[level] == t]) - {-1}) if below else []
            children = [below_labels.get(c, f"topic {c}") for c in children] if len(children) > 1 else []
            prompt_topics.append(describe(t, int(np.sum(at[level] == t)), words[t],
                                          central_papers(embeddings, at[level], t), children, data))
        named = labels.loc[labels['level'] == level, 'label'].tolist()
        print(f"Naming {len(todo[level])} {level} topics with {config.CLAUDE_MODEL}...")
        rows = name_level(level, todo[level], named, prompt_topics)
        new = pd.DataFrame({'level': level, 'topic': todo[level], 'label': [r['label'].strip() for r in rows],
                            'description': [r['description'].strip() for r in rows],
                            'keywords': [', '.join(words[t]) for t in todo[level]], 'label_source': 'ai',
                            'model': config.CLAUDE_MODEL, 'labelled_on': taxonomy.today()})
        labels = pd.concat([labels, new], ignore_index=True)
        labels['order'] = labels['level'].map(LEVELS.index)
        labels = labels.sort_values(['order', 'topic']).drop(columns='order')
        labels[taxonomy.LABEL_COLUMNS].to_csv(taxonomy.labels_path(major), index=False)  # saved level by level
        for row in new.itertuples():
            print(f"  {level} {row.topic}: {row.label}")


if __name__ == '__main__':
    main()
