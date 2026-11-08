"""Paths and settings shared by the pipeline scripts.

Paths are built from the repository root, so the scripts can be run from any folder.
"""
import configparser
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# Folders
DATA_DIR = ROOT / 'get_data'
NER_DIR = ROOT / 'NER'
MAPS_DIR = NER_DIR / 'maps'
PREPROCESSED_DIR = ROOT / 'bertopic' / 'preprocessed_data'
PAPER_MODEL_DIR = ROOT / 'bertopic' / 'model'  # the published paper's model; not used by the live pipeline
TAXONOMY_DIR = ROOT / 'bertopic' / 'taxonomy'  # versioned topics and paper assignments (see taxonomy.py)
MODELS_DIR = ROOT / 'bertopic' / 'models'  # pickled models of each major version: base.pkl, addition_<k>.pkl
EVALUATION_DIR = ROOT / 'bertopic' / 'evaluation'  # evaluate.py results (its cache/ subfolder is not committed)
GRAPHS_DIR = ROOT / 'graphs'
SITE_DIR = ROOT / 'site'  # the public website (GitHub Pages); export_site.py and newsletter.py write its data/
SITE_DATA_DIR = SITE_DIR / 'data'
NEWSLETTER_DIR = SITE_DATA_DIR / 'newsletter'  # one <date>.json per weekly issue, and index.json
LOGS_DIR = ROOT / 'logs'  # weekly.py logs, not committed

# Files shared between steps. Every per-paper file is keyed on `paper_id` (see paper_ids()), never on row order.
# ALLDATA is append-only: new papers are added at the end, and each later step only processes the ids it lacks.
SCOPUS_EXPORT = DATA_DIR / 'scopus_data.csv'  # raw SCOPUS export, not committed
SCOPUS_LAST_FETCH = DATA_DIR / 'scopus_last_fetch.txt'  # date (YYYYMMDD) of the last fetch_scopus.py run
ALLDATA = DATA_DIR / 'alldata.csv'
# Citation count of each paper and the date it was retrieved (paper_id, cited_by, retrieved_on). fetch_scopus.py adds
# the new papers, refresh_citations.py updates every paper. alldata.csv's `Cited by` is the count when first fetched.
CITATIONS = DATA_DIR / 'citations.csv'
# Date each paper was appended by fetch_scopus.py (paper_id, added_on); papers from before the record have no row.
# The weekly newsletter covers the papers of the latest date.
ADDED = DATA_DIR / 'added.csv'
EMBEDDINGS = PREPROCESSED_DIR / 'embeddings.npz'  # arrays `paper_id`, `embeddings` (float16) and `encoder`
BASIN_LABELS = NER_DIR / 'merged_fuzzy_classes_labels.csv'
BASIN_GRID_COUNTS = MAPS_DIR / 'range_area_counts.csv'

# Secrets (GeoNames username), not version controlled. See configurations.ini.example.
SECRETS = ROOT / 'configurations.ini'

# ---------------------------------------------------------------------------------------------
# Analysis settings
# ---------------------------------------------------------------------------------------------

# Citation analyses use the citations per year since publication, age = retrieval date - middle of the publication
# year, and only papers at least this many years old (younger papers have had too little time to be cited).
MIN_CITATION_AGE = 2
# Minimum papers of a topic x year cell for its Gini coefficient to be shown, and of a topic x basin cell for its median
# citation rate. MIN_GINI_PAPERS from evaluate.py's `sizes` stage: the 95% interval of a Gini coefficient of 50 papers is
# +-0.13 (about +-0.16 at 30), see bertopic/evaluation/CHOICE.md.
MIN_GINI_PAPERS = 50
MIN_CELL_PAPERS = 10

# SCOPUS search of the paper (it was run in three PUBYEAR slices; fetch_scopus.py adds a load-date filter instead)
SCOPUS_QUERY = ('TITLE-ABS-KEY ( ( "marine debris" OR "marine plastic debris" OR "marine plastic pollution" OR '
                '"plastic pollution" OR "marine plastic" OR "plastic debris" OR microplastic OR nanoplastic OR '
                'microfiber OR "plastic pellet" ) AND ( marine OR coast* OR sea OR ocean* OR *shore OR *pelagic OR '
                'benth* OR sediment ) )')

# Sentence-transformer used for the abstract embeddings. Qwen3-Embedding is instruction-tuned: the prompt (its format
# for symmetric tasks such as clustering) is prepended to every abstract. Its embeddings are truncated (Matryoshka) to
# EMBEDDING_DIM dimensions and L2-normalized, and saved as float16, which keeps embeddings.npz under GitHub's 100 MB.
# Changing any of these re-encodes every paper (embeddings.py), and a fitted taxonomy needs a re-baseline.
EMBEDDING_MODEL = 'Qwen/Qwen3-Embedding-4B'
EMBEDDING_DIM = 1024
EMBEDDING_PROMPT = ('Instruct: Identify the main research topic of the given scientific abstract on marine plastic '
                    'pollution\nQuery:')

# UMAP / HDBSCAN parameters of the topic model, chosen by hand with evaluate.py (UMAP from its `umap` stage, HDBSCAN
# from its `consensus` stage; record the choice in bertopic/evaluation/CHOICE.md). The UMAP seed is set per run by the
# consensus (consensus.py). UMAP: evaluate.py `umap` on the Qwen embeddings (2026-11-06).
UMAP_PARAMS = dict(n_components=5, n_neighbors=15, min_dist=0, metric='cosine')
# HDBSCAN and consensus: evaluate.py `report` and `consensus` (configuration 45, 2026-11-06).
HDBSCAN_PARAMS = dict(min_cluster_size=50, min_samples=5, metric='euclidean', cluster_selection_epsilon=0.15,
                      cluster_selection_method='eom')

# Consensus of the baseline (consensus.py): number of seeded runs, the share of runs in which papers must be clustered
# together, and the neighbours that vote on a new paper's topic
CONSENSUS_RUNS = 30
CONSENSUS_THRESHOLD = 0.5
CONSENSUS_KNN = 15

# Levels of the topic hierarchy: {name: minimum papers per topic}. The fine topics (minimum 0) are merged up until
# every topic of a level has its minimum, read off evaluate.py's `sizes` stage (bertopic/evaluation/CHOICE.md): at 150
# papers the 95% interval of a basin share is within +-0.10 and of a Gini coefficient +-0.085; at 400, +-0.06 and +-0.05.
TOPIC_LEVELS = {'fine': 0, 'mid': 150, 'coarse': 400}
# Level of the hierarchy of analysis.py's gini.csv (None: the fine topics). The site (export_site.py) has every level.
GINI_LEVEL = 'mid'


# Weekly assignment and topic additions (see assign.py and discover.py). The thresholds are set in the session on
# unmatched papers; while they are None, no re-baseline is flagged and discover.py does not run.
# Flag a re-baseline when the share of unmatched papers among those predicted since it exceeds the share expected of
# papers like the fitted ones (params.json `expected_unmatched`, from the leave-one-out vote) by more than this
REBASELINE_POOL_SHARE = None
NOVELTY_MAX_SIMILARITY = None  # a candidate topic is added if its centroid is less similar than this to every topic

# Topic search of the website: export_site.py embeds each topic's label, description and keywords with SEARCH_MODEL, and
# the browser embeds the visitor's query with SEARCH_MODEL_WEB (the same model as ONNX, run by transformers.js) after
# SEARCH_QUERY_PREFIX (the model's instruction for short queries). Small enough (~35 MB) to download in a browser.
SEARCH_MODEL = 'BAAI/bge-small-en-v1.5'
SEARCH_MODEL_WEB = 'Xenova/bge-small-en-v1.5'
SEARCH_QUERY_PREFIX = 'Represent this sentence for searching relevant passages: '

# Claude model of the AI-written texts (labels.py, newsletter.py), run headless through the `claude` CLI (ai.py)
CLAUDE_MODEL = 'claude-opus-5-5'
# A broad topic gets its own newsletter section with at least this many of the week's papers; the rest share one
NEWSLETTER_MIN_SECTION = 3

# Colour palette for topics and basins
COLORS = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#D55E00", "#0072B2", "#CC79A7",
          "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#00FFFF", "#FF00FF", "#C0C0C0",
          "#808080", "#800000", "#808000", "#008000", "#800080", "#008080"]


def reference_date():
    """Date of the last fetch_scopus.py run: the papers of alldata.csv are those loaded into Scopus up to it."""
    return dt.datetime.strptime(SCOPUS_LAST_FETCH.read_text().strip(), '%Y%m%d').date()


def last_full_year():
    """Last publication year complete at the last fetch: the end of the plots over time."""
    return reference_date().year - 1


def eids(links):
    """Scopus EID of each paper, from its `Link` column."""
    return links.str.extract(r'eid=([^&]+)', expand=False)


def paper_ids(data):
    """Key of each paper of an alldata-like table: its DOI, lower-cased and stripped, or `eid:<EID>` without a DOI."""
    doi = data['DOI'].str.strip().str.lower()
    return doi.fillna('eid:' + eids(data['Link'])).rename('paper_id')


def encoder():
    """The settings that determine the embeddings; saved with them, so that a change re-encodes every paper."""
    return f"{EMBEDDING_MODEL} | dim={EMBEDDING_DIM} | {EMBEDDING_PROMPT!r}"


def load_embeddings(ids=None, expect=None):
    """Embeddings of the papers `ids`, in that order (all saved embeddings and their ids when `ids` is None).

    Raises if they were not made by the encoder `expect` (by default the current settings, encoder()), so that a
    model never mixes embeddings of two encoders."""
    expect = expect or encoder()
    with np.load(EMBEDDINGS) as saved:
        made_by = str(saved['encoder']) if 'encoder' in saved else None
        if made_by != expect:
            raise ValueError(f"embeddings.npz was made by {made_by!r}, not {expect!r}. Run embeddings.py if the "
                             "settings changed; a model fitted on other embeddings needs a re-baseline.")
        saved_ids, embeddings = saved['paper_id'], saved['embeddings'].astype(np.float32)
    if ids is None:
        return saved_ids, embeddings
    position = {paper_id: i for i, paper_id in enumerate(saved_ids)}
    missing = [paper_id for paper_id in ids if paper_id not in position]
    if missing:
        raise KeyError(f"{len(missing)} papers have no embedding (e.g. {missing[:3]}); run embeddings.py")
    return embeddings[[position[paper_id] for paper_id in ids]]


def saved_encoder():
    """encoder() of the saved embeddings (None if there are none, or they predate the record)."""
    if not EMBEDDINGS.exists():
        return None
    with np.load(EMBEDDINGS) as saved:
        return str(saved['encoder']) if 'encoder' in saved else None


def save_embeddings(ids, embeddings):
    np.savez(EMBEDDINGS, paper_id=np.asarray(ids, dtype=str), embeddings=np.asarray(embeddings, dtype=np.float16),
             encoder=encoder())


def load_citations():
    """citations.csv indexed by paper_id: cited_by and retrieved_on (a datetime)."""
    return pd.read_csv(CITATIONS, parse_dates=['retrieved_on']).set_index('paper_id')


def save_citations(citations):
    citations = citations.copy()
    citations['retrieved_on'] = pd.to_datetime(citations['retrieved_on']).dt.strftime('%Y-%m-%d')
    citations.rename_axis('paper_id').reset_index().to_csv(CITATIONS, index=False)


def read_secrets():
    """Read configurations.ini at the repository root."""
    if not SECRETS.exists():
        raise FileNotFoundError(f"{SECRETS} not found; copy configurations.ini.example and fill it in.")
    secrets = configparser.ConfigParser()
    secrets.read(SECRETS)
    return secrets
