"""Step 6: the combined topic / ocean basin / citation analyses of the current taxonomy version.

Topics and labels come from the taxonomy (bertopic/taxonomy/, see taxonomy.py): every paper of alldata.csv must
have a topic (run assign.py), a basin row (run ner.py) and a citation count (get_data/citations.csv, see
refresh_citations.py). The topic words are recomputed over all current papers.

Citation analyses use each paper's citations per year since publication (citation_rates), for the papers at least
config.MIN_CITATION_AGE years old. Writes tables to bertopic/preprocessed_data/. The figures are drawn by the
website (site/) from the aggregates of export_site.py.
"""
import numpy as np
import pandas as pd

import config
import taxonomy

PRE = config.PREPROCESSED_DIR
CITES = 'Citations per year'


def citation_rates(data):
    """Citation count, retrieval date, age (years) and citations per year of each paper of `data` (which has
    `paper_id` and `Year`), aligned with `data`.

    age = retrieval date - middle of the publication year. The rate is NaN for papers younger than
    config.MIN_CITATION_AGE, which the citation analyses leave out."""
    citations = config.load_citations().reindex(data['paper_id'])
    if citations['cited_by'].isna().any():
        raise SystemExit(f"{citations['cited_by'].isna().sum()} papers have no citation count; "
                         "run refresh_citations.py")
    retrieved = citations['retrieved_on']
    retrieved_year = (retrieved.dt.year + (retrieved.dt.dayofyear - 1) / (365 + retrieved.dt.is_leap_year)).to_numpy()
    age = retrieved_year - (data['Year'].to_numpy() + 0.5)
    cited_by = citations['cited_by'].to_numpy(dtype=float)
    rate = np.where(age >= config.MIN_CITATION_AGE, cited_by / np.maximum(age, 1e-9), np.nan)
    return pd.DataFrame({'Citations': cited_by.astype(int), 'Retrieved': retrieved.dt.date.to_numpy(),
                         'Age': age.round(2), CITES: rate}, index=data.index)


def load_topic_model(docs, topics, embeddings):
    """The current base model, with its topic words, topic embeddings and representative documents recomputed for
    `topics` (which may include papers and topics the base model was not fitted on)."""
    topic_model = taxonomy.load_model(taxonomy.major(taxonomy.current_version()))
    topic_model.update_topics(docs, topics=topics, vectorizer_model=topic_model.vectorizer_model)
    documents = pd.DataFrame({"Document": docs, "ID": range(len(docs)), "Topic": topics, "Image": None})
    # Centroid topic embeddings, as in the fit. update_topics does not rebuild them for added topics.
    topic_model._create_topic_vectors(documents=documents, embeddings=embeddings)
    # Representative documents of each topic, with BERTopic's own (seeded) procedure
    topic_model._save_representative_docs(documents)
    return topic_model


def gini(x):
    """Gini coefficient of the citations (0 = equally cited, 1 = all citations in one paper), corrected for its
    small-sample bias by n / (n - 1). NaN for fewer than 2 papers or no citations."""
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n < 2 or x.sum() == 0:
        return np.nan
    # Equivalent to sum(|xi - xj|) / (2 n^2 mean(x)), computed on the sorted values
    return np.sum((2 * np.arange(1, n + 1) - n - 1) * x) / (n * x.sum()) * n / (n - 1)


def main():
    topic_labels = taxonomy.labels()

    # --- Data -------------------------------------------------------------------------------------
    data = pd.read_csv(config.ALLDATA)
    data['paper_id'] = config.paper_ids(data)
    assigned = data['paper_id'].map(taxonomy.load_assignments().set_index('paper_id')['topic'])
    if assigned.isna().any():
        raise SystemExit(f"{assigned.isna().sum()} papers have no topic; run assign.py")
    topics = assigned.astype(int).tolist()
    basins = pd.read_csv(config.BASIN_LABELS).set_index('paper_id')['Basin']
    if not data['paper_id'].isin(basins.index).all():
        raise SystemExit("Some papers have no basin row; run ner.py")
    cites = citation_rates(data)

    docs = data['Abstract'].tolist()
    timestamp = data['Year']
    classes = data['paper_id'].map(basins).fillna('Unknown').rename('Basin')
    embeddings = config.load_embeddings(data['paper_id'])
    data[['paper_id', 'Title', 'DOI', 'Authors']].to_csv(PRE / 'id.csv', index=False)

    # --- Topic model ----------------------------------------------------------------------------
    assert set(topics) <= set(topic_labels), "some assigned topics have no row in the taxonomy's topics.csv"
    topic_model = load_topic_model(docs, topics, embeddings)
    topic_ids = sorted(topic for topic in topic_model.get_topics() if topic != -1)
    print(f"{len(topic_ids)} topics (taxonomy {taxonomy.current_version()})")

    # Representative documents of each topic, with their paper_id, DOI and title (joined on topic and abstract, as a
    # few papers share an abstract)
    paper_info = (data.assign(Topic=topics).drop_duplicates(['Topic', 'Abstract'])
                  [['Topic', 'Abstract', 'paper_id', 'DOI', 'Title']])
    rep_docs = pd.DataFrame([(topic, rank, doc)
                             for topic, topic_docs in topic_model.get_representative_docs().items()
                             for rank, doc in enumerate(topic_docs, 1)],
                            columns=['Topic', 'Rank', 'Abstract'])
    rep_docs = (rep_docs.merge(paper_info, on=['Topic', 'Abstract'], how='left')
                [['Topic', 'Rank', 'paper_id', 'DOI', 'Title', 'Abstract']].sort_values(['Topic', 'Rank']))
    rep_docs.to_csv(PRE / 'rep_docs.csv', index=False)

    topic_model.set_topic_labels({t: topic_labels[t] for t in [-1, *topic_ids] if t in topic_labels})

    # --- Topics over time -----------------------------------------------------------------------
    # evolution_tuning only writes into a temporary copy in BERTopic (no effect), at a large cost: off
    topics_over_time = topic_model.topics_over_time(docs, timestamp, evolution_tuning=False)
    topics_over_time.to_csv(PRE / 'topics_over_time.csv')

    # --- Paper table: topic, basin, year and citations of each paper ----------------------------
    papers = pd.concat([data[['paper_id', 'Title', 'DOI', 'Year']],
                        pd.DataFrame({'Topic': topics, 'Topic names': pd.Series(topics).map(topic_labels)}),
                        classes, cites], axis=1)
    for level in sorted(config.TOPIC_LEVELS, key=config.TOPIC_LEVELS.get, reverse=True):
        if config.TOPIC_LEVELS[level]:  # the merged levels (the fine topics are `Topic`)
            papers[f'Topic ({level})'] = taxonomy.level_topics(topics, level)
    papers.assign(Abstract=data['Abstract']).to_csv(PRE / 'docs_topics.csv', index=False)
    freq = topic_model.get_topic_info()
    freq.to_csv(PRE / 'freq.csv')

    # Topic frequency table: size, full name and mean citations per year of each topic
    topic_frequency = freq.loc[freq['Topic'] != -1, ['Topic', 'Count']].copy()
    topic_frequency['Topics'] = topic_frequency['Topic'].map(topic_labels)
    topic_frequency['Average citation'] = topic_frequency['Topic'].map(papers.groupby('Topic')[CITES].mean())
    topic_frequency.to_csv(PRE / 'topic_frequency.csv')

    # --- Gini coefficients of the citations per year --------------------------------------------
    # at the level config.GINI_LEVEL of the topic hierarchy (None: the fine topics)
    level = config.GINI_LEVEL
    gini_papers = papers.assign(Topic=papers[f'Topic ({level})'] if level else papers['Topic'])
    gini_papers = gini_papers[gini_papers[CITES].notna()]
    print("Gini coefficient of all papers:", gini(gini_papers[CITES]))
    print("Mean Gini coefficient of the topics:", gini_papers.groupby('Topic')[CITES].apply(gini).mean())

    # Every topic x year cell is saved; the site shows those with at least config.MIN_GINI_PAPERS papers, as the
    # coefficient of a few papers is imprecise
    gini_n = (gini_papers.groupby(['Topic', 'Year'])[CITES].agg(['size', gini]).reset_index()
              .rename(columns={'size': 'Papers', 'gini': 'Gini Coefficient'}))
    gini_n.to_csv(PRE / 'gini.csv')

if __name__ == '__main__':
    main()
