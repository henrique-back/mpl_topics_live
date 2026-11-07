# Known issues

Open reproducibility gaps in this repository. Each one needs action from the authors; none is fixed by code alone.

## 1. The choice of hyperparameters cannot be traced

`pipeline/config.py` sets these UMAP / HDBSCAN parameters:

| UMAP `n_components` | UMAP `n_neighbors` | HDBSCAN `min_cluster_size` | HDBSCAN `min_samples` | HDBSCAN `cluster_selection_epsilon` |
|---|---|---|---|---|
| 11 | 8 | 47 | 2 | 0.15 |

Nothing in the repository shows where they came from:

- The search cells of `hyperparameter_tuning.ipynb` (later `pipeline/tune.py`, now replaced by `pipeline/evaluate.py`) were never executed in the committed version of the paper repository, so the notebook had no saved outputs.
- Its output, `bertopic/preprocessed_data/tuning_results.csv`, is not committed.

Re-running the search elsewhere will not recover the original ranking, because UMAP is not reproducible across machines and library versions even with `random_state=42`. For example, refitting the saved model's exact configuration with the versions in `requirements.txt` on macOS gave 38 topics instead of 39. Only 49% of the papers kept their topic (adjusted Rand index 0.54).

**To close:** this can't be closed for the paper's model: the environment that produced `bertopic/model` and its search are gone.

**In the live review:** the live taxonomy v1 (`bertopic/models/v1/`) was fitted with these same parameters, so its parameter choice is also untraced. `params.json` records them. From v2 on, a re-baseline is traceable: `pipeline/evaluate.py` writes its evaluation to `bertopic/evaluation/` (committed), including the `consensus` stage that evaluates the consensus itself (threshold, neighbours of the vote, number of runs); the choice and its reason are recorded in `bertopic/evaluation/CHOICE.md`; and `baseline.py` fits a consensus over many seeds, whose split-half agreement, encoder, package versions and git commit are saved in `params.json`.

## 2. Figures that no code produces

No script writes these files:

| File | Note |
|---|---|
| `graphs/macro_vs_micro_topics_over_time.html` / `.svg` | Added in the "2025 data" commit; the code that made them is not in the repository. |
| `graphs/top_topics_over_time.html` / `.svg` | Same. |
| `graphs/cumulative_citations_selected.svg` | Exported by hand from a figure of selected topics of the paper. The live analysis shows these curves on the website (`site/`), for every topic. |
| `graphs/selected_gini_logN.svg` | Same. |
| `graphs/selected_topics_over_time.svg` | Same. |

The hand-exported `.svg` files show the paper's topics and were **not** regenerated when the analysis was updated; `pipeline/analysis.py` no longer draws figures (the website does). `graphs/map.svg` is not affected: `pipeline/basin_map.py` writes it.

**To close:**
- Commit the code for the `macro_vs_micro` and `top_topics_over_time` figures.
- Export the `.svg` files from code with `fig.write_image(...)`, which requires the `kaleido` package, instead of by hand.

## 3. Re-running entity extraction changes the locations

`NER/data_loc.csv` was produced in August 2025. Re-running the `extract` stage of `pipeline/ner.py` with the versions in `requirements.txt` gives a different set of locations for some papers: 6 of the first 40 differ. For example, "N Spain" instead of "Spain", and species such as *Posidonia oceanica* tagged as places. The extraction code is unchanged, so the difference comes from the spaCy / model environment.

A full re-run would therefore change the basin of some existing papers.

**Mitigated:** `pipeline/ner.py` now extracts locations only for new papers and keeps the committed rows, so existing papers keep their basin. New papers are still processed in the current environment. To close fully, pin the environment that produced `data_loc.csv` and record it.

## 4. Taxonomy v1 can't assign new papers

The v1 base model (`bertopic/models/v1/base.pkl`) was fitted on 768-dimensional SPECTER embeddings of the abstracts as exported, copyright statements included. Since November 2026:

- `clean_data.normalize_abstract` removes the copyright statement that nearly all export abstracts (98.9%) end with, such as "© 2024 Elsevier Ltd.", and that the Search API's abstracts (`fetch_scopus.py`) almost never have. Under v1 the statement mattered: in a test on 600 old papers, removing it changed the v1 topic of 6.3% of them (1% from re-running the projection alone), mostly moves to or from the outliers, although the SPECTER embedding barely moved (median cosine similarity 0.9986).
- The embeddings are made with Qwen3-Embedding-4B (1024 dimensions) instead of SPECTER, which has been deleted.

v1's UMAP can't project the new embeddings, so `assign.py` can't run under v1, and its topic centroids (`analysis.py`, `discover.py`) would mix two embedding spaces.

**To close:** re-baseline (`baseline.py`) on the Qwen embeddings of the normalized abstracts. v1 stays frozen as the record of the first taxonomy.

## 5. Citation counts of the paper's export can't be dated

The paper's analysis divided citation counts by `2024 - publication year`, but the export's counts were taken on an unrecorded date, and the live review's new papers had their counts taken when fetched. The two sets were not comparable.

**Mitigated:** citation counts now live in `get_data/citations.csv` with the date each was retrieved, `pipeline/refresh_citations.py` refreshes them all from Scopus, and ages are measured from each paper's retrieval date. The counts used in the published paper can't be redated, so the paper's citation figures can't be reproduced from the live data.

## 5. The basin map of `basin_map.py` is shifted by half a cell

The values of the basin grid (`NER/range_area.csv`) are the centres of 1° cells (x.5 degrees), but `basin_map.basin_polygons` draws each cell from its value to the value + 1°, so `graphs/map.svg` is shifted half a degree north-east, and `fill_grid_gaps` fills the gaps this leaves on the equator and the 0° meridian. The website's basin map (`export_site.export_geo`) centres the cells and fills the missing grid cells by nearest cell instead. The classification of papers (`ner.py`) uses the grid points themselves and is not affected.

**To close:** centre the cells in `basin_map.py` as in `export_site.py`.

## 6. AI-written text is published without review

The topic labels and descriptions (`pipeline/labels.py`) and the weekly newsletter (`pipeline/newsletter.py`) are written by Claude and published as generated; the site says so. `newsletter.py` removes links to papers that were not in the week's input, but nothing checks the statements against the abstracts. Claude's output also changes from run to run, so `--redo` gives different text.

**To close (if needed):** review the labels after each re-baseline (edit `labels.csv`; they are only regenerated with `--redo`), and read each issue before the push, or sample issues against their abstracts.

## 7. Scopus access from the scheduled machine

On 2026-11-07, from the machine that runs the weekly update, the Scopus API refused `fetch_scopus.py` (401 `AUTHORIZATION_ERROR` for the COMPLETE view) and `refresh_citations.py` (400 "Exceeds the maximum number allowed for the service level"). Institutional tokens are usually tied to the institution's network, so the scheduled run needs that network (or a VPN) and a valid `insttoken`.

**To close:** confirm the key and token with Elsevier / the library, and run `pipeline/weekly.py --dry-run` once from the scheduled machine.
