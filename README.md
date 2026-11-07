# mpl_topics_live

## Overview

A living review of the scientific literature on microplastics, to be updated with new papers every week. It is a fork of [mpl_topics](https://github.com/henrique-back/mpl_topics), the code of a paper currently under consideration for publication; that repository keeps the version used in the paper.

Abstracts retrieved from SCOPUS are grouped into research topics with [BERTopic](https://maartengr.github.io/BERTopic/), and each paper is assigned to an ocean basin using named-entity recognition (NER) on the locations mentioned in its abstract. The two classifications are then combined with publication dates and citation counts.

The code relies on publicly available libraries, notably scikit-learn, BERTopic and spaCy.

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [Data](#data)
4. [Ocean basin classification (NER)](#ocean-basin-classification-ner)
5. [Topic modelling](#topic-modelling)
6. [Results](#results)

## Installation

Requires **Python 3.12 or newer**. The code runs on Windows, Linux and macOS. On macOS, PyTorch needs an Apple Silicon Mac running macOS 14 or later.

Create a virtual environment and install the pinned dependencies, which include the spaCy model `en_core_web_trf`:

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

```powershell
# Windows (PowerShell)
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The exact versions used are listed in `requirements.txt`. Newer pandas (3.x) and matplotlib (3.9+) releases remove functions the pipeline uses, so keep the pinned versions.

## Usage

The pipeline is a set of Python scripts in `pipeline/`. The weekly update runs them in this order, from any folder (`python pipeline/weekly.py` runs them all; see [Weekly schedule](#weekly-schedule)):

| Step | Script | What it does | Main output |
|---|---|---|---|
| 1 | `python pipeline/fetch_scopus.py` | Fetches the papers added to SCOPUS since the last run, cleans them (removes duplicate DOIs, missing abstracts and corrigenda) and appends them, with their citation counts | `get_data/alldata.csv`, `get_data/citations.csv` |
| 2 | `python pipeline/embeddings.py` | Pre-computes the embeddings of the new abstracts with the `Qwen/Qwen3-Embedding-4B` sentence-transformer (1024 dimensions) | `bertopic/preprocessed_data/embeddings.npz` |
| 3 | `python pipeline/ner.py` | Extracts the locations of the new papers, geocodes the new locations and assigns each paper an ocean basin | `NER/merged_fuzzy_classes_labels.csv` |
| 4 | `python pipeline/assign.py` | Places the new papers in the topics of the current taxonomy | `bertopic/taxonomy/v<N>/assignments.csv`, `bertopic/taxonomy/runs.csv` |
| 5 | `python pipeline/basin_map.py` | Maps the number of papers per ocean basin | `graphs/map.svg`, `graphs/map.png` |
| 6 | `python pipeline/analysis.py` | The combined topic / basin / citation tables of the current taxonomy | `bertopic/preprocessed_data/` |
| 7 | `python pipeline/labels.py` | Names and describes, with Claude, the topics that have no AI label yet (only after a new taxonomy version) | `bertopic/taxonomy/v<N>/labels.csv` |
| 8 | `python pipeline/layout.py` | Places the new papers on the 2-D paper map of the website | `bertopic/models/v<N>/layout.csv` |
| 9 | `python pipeline/newsletter.py` | Writes, with Claude, the weekly newsletter on the papers of the latest fetch | `site/data/newsletter/` |
| 10 | `python pipeline/export_site.py` | Exports the aggregates the website shows | `site/data/` |

Occasional steps (see [Topic modelling](#topic-modelling)):

| Script | When | What it does |
|---|---|---|
| `python pipeline/refresh_citations.py` | Monthly (`weekly.py` runs it on the first Monday of the month), and after a full rebuild | Updates the citation count of every paper from SCOPUS |
| `python pipeline/evaluate.py <stage>` | Before a re-baseline | Chooses the topic sizes, UMAP, HDBSCAN and consensus parameters by evaluation (stages `sizes`, `umap`, `search`, `report`, `consensus`) |
| `python pipeline/baseline.py` | First live model, and when `assign.py` flags a re-baseline | Fits a new (consensus) topic model and opens a new major version of the taxonomy |
| `python pipeline/discover.py` | Monthly or quarterly (not usable yet) | Adds the topics that have formed among the unmatched papers |

Every intermediate file is included in the repository, so any step can be run on its own without repeating the earlier ones. Paths and analysis settings (minimum citation age, model parameters, thresholds, the Claude model) are in `pipeline/config.py`; topic labels are in the taxonomy (`bertopic/taxonomy/v<N>/labels.csv`, with the keyword labels in `topics.csv`).

Open reproducibility gaps are listed in [KNOWN_ISSUES.md](KNOWN_ISSUES.md).

## Data

The complete dataset, `get_data/alldata.csv` (about 20,000 papers from 1977 on), is stored in this repository, so the analysis can be reproduced without SCOPUS access. The first 12,398 papers come from a SCOPUS export of 2025; the rest were added by the weekly updates. The citation count of each paper, with the date it was retrieved, is in `get_data/citations.csv`.

Retrieving the data from SCOPUS requires an Elsevier API key and an institutional token, set in the `[scopus]` section of `configurations.ini` (see `configurations.ini.example`).

### Weekly updates

The dataset is updated incrementally, and each step only processes what it has not processed before:

1. `pipeline/fetch_scopus.py` runs the paper's search (`config.SCOPUS_QUERY`) restricted to the records first loaded into Scopus after the last fetch (`ORIG-LOAD-DATE`, with a 7-day overlap). The date of the last fetch is kept in `get_data/scopus_last_fetch.txt`; `--since YYYYMMDD` overrides it. Fetched papers are cleaned like the original export; papers already in `alldata.csv` (same EID or DOI) and papers without a DOI are skipped, and the rest are **appended** to `alldata.csv`.
   The citation count of each new paper is added to `get_data/citations.csv`, dated to the fetch, and the date it was added to `get_data/added.csv` (the papers of the weekly newsletter). `pipeline/refresh_citations.py` updates the counts of all papers (run it monthly).
2. `pipeline/embeddings.py` encodes only the papers that have no saved embedding yet.
3. `pipeline/ner.py` runs spaCy only on the papers that have no row in `NER/data_loc.csv` yet, and sends GeoNames only the locations it has never looked up (found ones are in `fuzzy_location_data.csv`, unknown ones in `fuzzy_error_locations.csv`). Progress is saved as it goes, so if the GeoNames credit limit is reached, re-run it later to continue.

4. `pipeline/assign.py` places only the papers without a topic, and leaves every other paper's topic unchanged.

Every per-paper file is keyed on `paper_id`: the DOI in lower case, or `eid:<Scopus EID>` for the one paper without a DOI (`config.paper_ids`). No file depends on the row order of `alldata.csv`. To recompute a step from scratch, delete its output (`embeddings.npz`, `data_loc.csv`, or the geocoding files).

### Full rebuild

To rebuild the dataset from a complete export instead (this invalidates the existing embeddings, basin classes and topic model):

1. Run the SCOPUS search and export the results as CSV to `get_data/scopus_data.csv`.
2. Run `pipeline/clean_data.py` to clean the export and write `get_data/alldata.csv`, then `pipeline/refresh_citations.py` to date the citation counts.
3. Re-run every later step. Papers kept from before keep their embeddings, locations and topics, since these are keyed on `paper_id`.

## Ocean basin classification (NER)

`pipeline/ner.py` assigns each paper to an ocean basin in four stages:

1. **Entity extraction:** spaCy's `en_core_web_trf` model extracts geographic entities (`GPE` and `LOC`) from every abstract.
2. **Geocoding:** each location is looked up with the [GeoNames](https://www.geonames.org/) search API. Locations that fail an exact search are retried with fuzzy search. Each location is looked up only once: locations GeoNames does not know are recorded and not sent again (remove one from `NER/fuzzy_error_locations.csv` to retry it), while locations hit by an API or network error are retried on the next run. Locations that GeoNames cannot find or places wrongly are given coordinates by hand (`ERROR_DICT` in `ner.py`). The requests are throttled to respect the API's rate limits.
3. **Basin assignment:** each location is matched to the nearest cell of a 1° ocean-basin grid (`NER/range_area.csv`, whose basin codes are named in `NER/range_basin_list.csv`). The match uses great-circle (haversine) distance. Only locations without a basin yet are matched, so existing assignments stay stable: when a location is exactly equidistant from two grid cells, scikit-learn versions break the tie differently.
4. **Paper classification:** each paper receives the basin that occurs most often among its located places, and coastal and open-ocean variants of the same basin are merged. Names that could not be geocoded do not take part in the vote. Papers without any located place are classified as `Unknown`.

Stages 1 and 2 are slow (stage 2 calls an external API). `python pipeline/ner.py --from geocode` skips stage 1, and `--from manual` runs only the offline part (manual coordinates, stages 3 and 4).

The GeoNames API needs a free account. The script reads the username from `configurations.ini` at the repository root, which is excluded from version control. Create it from the template before geocoding:

```bash
cp configurations.ini.example configurations.ini   # then set your username
```

## Topic modelling

1. **Generate embeddings:** `pipeline/embeddings.py` pre-computes the embeddings to speed up topic modelling. The stored embeddings are then passed to BERTopic when the model is built.
2. **Choosing the model:** `pipeline/evaluate.py` separates three choices that a single score would mix up. Results go to `bertopic/evaluation/`, and the chosen values are set by hand in `pipeline/config.py`, with the reasons recorded in `bertopic/evaluation/CHOICE.md`.
   - `sizes`: how many papers a topic needs for each analysis to be precise enough (the confidence interval of its Gini coefficient and basin shares, its papers per year). This sets the levels of the topic hierarchy, not the number of topics.
   - `umap`: the UMAP representation, by how well it preserves each paper's neighbours in the embedding space, and how stable that is across seeds.
   - `search` and `report`: the HDBSCAN parameters, fitted on 10 UMAP seeds and on held-out folds. Each configuration is scored on a density-based validity index that penalizes outliers, the coherence and diversity of its topic words, the stability of its topics across seeds, how new papers are placed, and how much its topics say about the authors' keywords. The report shows the configurations that no other beats on all of stability, validity, keyword information and held-out outliers.
   - `consensus`: the consensus that `baseline.py` actually fits, for the configurations of the report: its topics, outliers and agreement between two halves of the seeds at each co-association threshold, and how its nearest-neighbour vote places papers (`CONSENSUS_THRESHOLD`, `CONSENSUS_KNN`, `CONSENSUS_RUNS`).
3. **Consensus and hierarchy:** `pipeline/baseline.py` fits the chosen configuration with 30 seeds and keeps the papers that are clustered together in most of them (a consensus, `pipeline/consensus.py`), so the topics don't depend on one seed. These fine topics are then merged, smallest first, into their most similar topic until each level of the hierarchy (`TOPIC_LEVELS`) has topics large enough for its analyses; `merges.csv` records the similarity of each merge. `params.json` records the parameters, the agreement between the two halves of the seeds, the share of papers the vote leaves unmatched, the encoder, the package versions and the git commit.
4. **Analysis:** `pipeline/analysis.py` writes the per-paper and per-topic tables (topic at each level, basin, citations per year, Gini coefficients at `GINI_LEVEL`). `pipeline/export_site.py` computes the aggregates of the website at every level of the hierarchy: topics over time, topics per ocean basin, median citations per year, Gini coefficients and cumulative citation curves.

### A living taxonomy

Refitting the topic model every week would change past results: UMAP results differ across machines and library versions even with a fixed seed, and a refit of the same configuration agrees with the paper's model at an adjusted Rand index of only 0.54. The live review therefore keeps its topics fixed between versioned updates (`pipeline/taxonomy.py` describes the files in `bertopic/taxonomy/` and `bertopic/models/`):

- **Major versions (v1, v2, ...)** are fits of the topic model on all papers (`baseline.py`). The fitted model is saved as a pickle, so new papers can be placed into the same topics. Each major version comes with a crosswalk from the previous one (`crosswalk.csv`: topics matched by the overlap of their papers, if their Jaccard index is at least 0.1, with the topics that split or merged). For v1 the previous version was the paper's model in `bertopic/model`, whose hand-written topic names are kept in `bertopic/model/labels.csv`; the live pipeline does not otherwise use it.
- **Weekly**, `assign.py` gives each new paper the most common topic among its 15 nearest papers in the embedding space, where the outliers vote as unmatched (v1, a single fit, still uses its UMAP and HDBSCAN's `approximate_predict`). Unmatched papers have topic -1. Past assignments never change.
- **Minor versions (v1.1, v1.2, ...)** add topics that have formed among the unmatched papers (`discover.py`). A topic model is fitted on the unmatched papers, and each cluster that is not too similar to an existing topic becomes a new topic. Unmatched papers of any age in that cluster move into it. Later papers reach the added topics through the addition's own model.
- **Re-baselines:** `assign.py` logs the share of unmatched papers among those placed since the baseline (`runs.csv`) and flags a re-baseline when it exceeds the share expected of papers like the fitted ones (`expected_unmatched` in `params.json`) by more than `REBASELINE_POOL_SHARE`. A re-baseline is run by hand with `baseline.py`.

The definition of the pool of unmatched papers and the thresholds (`REBASELINE_POOL_SHARE`, `NOVELTY_MAX_SIMILARITY`) are not set yet, so no re-baseline is flagged and `discover.py` stops with a message.

**Topic labels** are written by Claude (`pipeline/labels.py`, through the `claude` command-line tool, so it uses the Claude Code login of the machine and needs no API key): a short plain-language name and a one- or two-sentence description of every topic at each level, from the topic's top c-TF-IDF words, its five papers closest to the centroid and, above the detailed level, the names of its subtopics. They are saved in `bertopic/taxonomy/v<N>/labels.csv` with the keywords, and used by every output through `taxonomy.labels()`; the keyword labels (top four c-TF-IDF words) stay in `topics.csv`. `analysis.py` recomputes the topic words and the three most representative documents of each topic over all current papers, and saves them to `bertopic/preprocessed_data/rep_docs.csv` with their `paper_id`, DOI and title. `bertopic/preprocessed_data/docs_topics.csv` has one row per paper: its topic at each level, basin, year and citations.

`bertopic/preprocessed_data/topic_frequency.csv` lists each topic's number of papers, label and mean citations per year.

**Citations per year:** to compare papers of different ages, citation counts are divided by the paper's age when they were retrieved:

```
citations per year = citations / (retrieval date - middle of the publication year, in years)
```

Papers younger than `MIN_CITATION_AGE` (2 years) have had too little time to be cited and are left out of the citation analyses. The Gini coefficients are corrected for their small-sample bias (by n / (n - 1)), and the website shows only topic-year cells with at least `MIN_GINI_PAPERS` papers. The charts over time stop at the last full year before the last fetch (`config.last_full_year()`).

## Website

The results are published as a static website, `site/` (plain HTML and JavaScript with [Apache ECharts](https://echarts.apache.org) from a CDN, no build step), served by GitHub Pages: `.github/workflows/pages.yml` publishes `site/` on every push to `main` that changes it (repository settings: Pages, Source: GitHub Actions). It has three tabs:

- **This week:** the weekly newsletter, written by Claude (`pipeline/newsletter.py`) from the abstracts of the papers added by the latest fetch: an overview, then a section per broad topic with at least `NEWSLETTER_MIN_SECTION` new papers, linking to a few of them by DOI; the other papers share a last paragraph. Links to DOIs that are not among the week's papers are removed. Issues are kept in `site/data/newsletter/` and listed as an archive. They are published without review, with a notice that they are AI-generated.
- **Explore the research:** a switch between the three levels of the topic hierarchy, a sunburst of the hierarchy and a topic selector; the selected topic is highlighted in every chart (the papers' 2-D map, topics over time, ocean basin map and topics per basin, topic x basin heatmap, median citations per year, cumulative citation curves, Gini coefficients over time) and can be shared by its address (`#explore/<level>/<topic>`).
- **Methods:** the pipeline in plain language, the stability of the topics (from `params.json` and `runs.csv`), the versions of the taxonomy and the limitations.

The site reads only the files of `site/data/`, written by `export_site.py`: aggregates, and for the paper map the coordinates and topic of each paper, without ids or titles. To look at it locally: `python -m http.server -d site`, then open http://localhost:8000.

The paper map uses a 2-D UMAP fitted once per major version of the taxonomy (`layout.py`); later papers are placed at the mean position of their nearest fitted papers, so the map keeps its layout until the next re-baseline.

### Weekly schedule

`pipeline/weekly.py` runs the weekly steps in order (and `refresh_citations.py` on the first Monday of the month), logging to `logs/weekly-<date>.log`. If every step succeeds, it commits the outputs and pushes, which publishes the site; if one fails, it commits nothing and shows a macOS notification. A re-baseline flagged by `assign.py` only raises a notification. `--dry-run` runs the steps without committing; `--from <step>` resumes at a step.

It is started every Monday at 06:00 by launchd (or at the next wake if the Mac was asleep). The Mac needs the `claude` command logged in, push access to GitHub without a password prompt (an SSH key in the keychain) and network access to Scopus. To install the schedule:

```bash
sed -e "s|__REPO__|$PWD|g" -e "s|__HOME__|$HOME|g" scheduling/br.ufba.mpl-weekly.plist > ~/Library/LaunchAgents/br.ufba.mpl-weekly.plist
mkdir -p logs
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/br.ufba.mpl-weekly.plist
launchctl kickstart gui/$(id -u)/br.ufba.mpl-weekly   # optional: run it now
```

To remove it: `launchctl bootout gui/$(id -u)/br.ufba.mpl-weekly`.

## Results

The main results are on the website (see [Website](#website)). The `graphs` folder keeps the ocean basin map (`map.svg`, `map.png`, from `basin_map.py`) and figures exported by hand for the paper (see [KNOWN_ISSUES.md](KNOWN_ISSUES.md)).
