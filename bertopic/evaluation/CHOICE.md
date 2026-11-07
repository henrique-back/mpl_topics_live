# Parameter choice for taxonomy v2

**Status: provisional.** Chosen on 2026-11-06 by applying the rules of `pipeline/evaluate.py` to its results, without a review by the authors. Data: 20,219 papers, Qwen3-Embedding-4B (1024 dimensions), normalized abstracts.

## UMAP (`umap` stage, `umap.csv`)

`UMAP_PARAMS = dict(n_components=5, n_neighbors=15, min_dist=0, metric='cosine')`

The rule takes the most seed-stable setting within 5% of the best kNN recall (0.340):

| n_neighbors | n_components | kNN recall | trustworthiness | seed stability |
|---|---|---|---|---|
| **15** | **5** | 0.325 | 0.977 | **0.609** |
| 10 | 10 | 0.337 | 0.976 | 0.607 |
| 10 | 25 | 0.340 | 0.977 | 0.600 |

This is close to a tie with n_neighbors=10, n_components=10: their seed stability differs by 0.002, within the noise of 3 seeds, and that setting has a slightly better kNN recall. The 5-dimensional setting was kept because it follows the rule, it is BERTopic's default, and density-based clustering works better in fewer dimensions. Larger n_neighbors (30, 50) lose kNN recall with no gain in stability.

## HDBSCAN (`search`, `report` stages)

`HDBSCAN_PARAMS = dict(min_cluster_size=50, min_samples=5, cluster_selection_epsilon=0.15)` (configuration 45 of `search.csv`)

31 of the 56 configurations are on the Pareto front of stability, DBCV, keyword information and held-out outliers (`pareto.csv`). The single-fit search alone does not single one out, so the choice was made on the consensus (next section). Configuration 45 is on the front, with the highest share of stable topics of all configurations (0.637). Its other single-fit scores: 54 topics, 10.1% outliers, stability AMI 0.954, held-out outliers 15.2%.

## Consensus (`consensus` stage, `consensus.csv`, `consensus_selected.csv`, `convergence.csv`)

`CONSENSUS_RUNS = 30`, `CONSENSUS_THRESHOLD = 0.5`, `CONSENSUS_KNN = 15` (unchanged)

Consensus of 30 runs at threshold 0.5, the best configurations by split-half reproducibility:

| config | mcs | ms | ε | topics | outliers | split ARI | split AMI | outlier kappa | best-match Jaccard | LOO kept (k=15) | LOO unmatched |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **45** | 50 | 5 | 0.15 | 55 | 10.1% | **0.932** | 0.989 | **0.900** | **0.940** | 96.3% | 4.9% |
| 37 | 40 | 5 | 0.15 | 59 | 9.6% | 0.928 | 0.989 | 0.885 | 0.935 | 96.2% | 4.3% |
| 29 | 30 | 5 | 0.15 | 64 | 9.4% | 0.931 | 0.987 | 0.891 | 0.922 | 96.1% | 3.9% |
| 39 | 40 | 10 | 0.15 | 69 | 14.4% | 0.888 | 0.997 | 0.887 | 0.942 | 96.0% | 7.9% |

- **Configuration:** 45 has the highest split-half ARI and best-match Jaccard with a low outlier share, and the largest minimum topic size (51) of the leading configurations. Configurations with `min_samples` 1–2 reproduce far less well (ARI about 0.71); those with ε = 0 leave 19–31% of papers as outliers (18.6% at best).
- **Threshold:** for configuration 45, 0.5 maximizes split-half ARI and outlier kappa. 0.4 merges less reliably (ARI 0.909); 0.6 and 0.7 raise the outlier share (11.7%, 13.9%) and lower ARI.
- **Neighbours:** k = 15 keeps 96.3% of clustered papers in their topic and leaves 4.9% unmatched. Smaller k keeps slightly more but leaves more unmatched; larger k keeps fewer.
- **Runs:** split-half agreement with 5 / 10 / 15 / 20 runs per half: ARI 0.835 / 0.929 / 0.932 / 0.944. It levels off by 10–15 runs per half, so 30 runs is on the plateau.

The leave-one-out vote is scored against a consensus that saw those papers. It tests the assignment rule, not how well the topics generalize.

## Topic sizes (`sizes` stage, `sizes.csv`)

`TOPIC_LEVELS = {'fine': 0, 'mid': 150, 'coarse': 400}`, `MIN_GINI_PAPERS = 50`

95% intervals from 1,000 random subsets of the corpus (Gini coefficients of the papers old enough for the citation analyses; basin shares of the five largest basins, of which about 54% of papers have one):

| papers | Gini half-width | largest basin-share half-width | papers per year (median of the last 10 years) |
|---|---|---|---|
| 25 | 0.173 | 0.250 | 2.0 |
| 50 | 0.130 | 0.163 | 4.0 |
| 100 | 0.100 | 0.115 | 8.0 |
| **150** | 0.085 | **0.097** | 12.1 |
| 200 | 0.070 | 0.081 | 16.1 |
| **400** | 0.054 | **0.058** | 32.2 |
| 1000 | 0.034 | 0.033 | 80.4 |

- **Levels:** `mid` topics (at least 150 papers, for the Gini and cumulative-citation analyses) give basin shares within ±0.10 and topic Gini coefficients within ±0.085; `coarse` topics (at least 400, for the basin analyses) give basin shares within ±0.06. The baseline merged the 55 fine topics into 33 `mid` and 16 `coarse` topics.
- **Gini cells:** the Gini-over-time plots show topic × year cells of `mid` topics. With at least 50 eligible papers a cell's Gini is within ±0.13, and 81 cells of 23 topics remain (years up to 2024; papers from 2025 are under `MIN_CITATION_AGE`). 30 papers would keep 126 cells of 30 topics at about ±0.16 (interpolated), about a quarter of a typical Gini (0.6); 100 papers only 39 cells of 10 topics.
- **Bias:** the bias-corrected Gini is still slightly low in small samples of these skewed rates: a mean of 0.594 at 50 papers against 0.608 at 1,000.
