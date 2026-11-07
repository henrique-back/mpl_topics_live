"""2-D layout of the papers for the site's paper map, stable within a major version of the taxonomy.

The first run of a major version fits a seeded 2-D UMAP of the papers the base model was fitted on (assignments.csv
`method == 'fit'`) and saves their coordinates. Later papers are placed at the mean position of their
config.CONSENSUS_KNN nearest fitted papers in the embedding space and appended, so existing points never move and the
map only changes layout at a re-baseline. No UMAP model is saved (it would hold all the embeddings).

Writes bertopic/models/v<N>/layout.csv: paper_id, x, y, method (fit | placed).
"""
import numpy as np
import pandas as pd
from umap import UMAP

import config
import taxonomy

# min_dist > 0 spreads the papers of a topic, so the map shows them (clustering uses min_dist 0, which packs them)
UMAP_2D = dict(n_neighbors=15, n_components=2, min_dist=0.25, metric='cosine', random_state=0)


def layout_path(major_version=None):
    return config.MODELS_DIR / (major_version or taxonomy.major(taxonomy.current_version())) / 'layout.csv'


def load_layout(major_version=None):
    return pd.read_csv(layout_path(major_version))


def main():
    major = taxonomy.major(taxonomy.current_version())
    path = layout_path(major)
    assignments = taxonomy.load_assignments(major)
    expect = taxonomy.load_params(major).get('encoder')  # the encoder of the fit, as consensus models require

    if path.exists():
        layout = pd.read_csv(path)
    else:
        fitted = assignments.loc[assignments['method'] == 'fit', 'paper_id']
        print(f"Fitting the 2-D layout of {major} on {len(fitted)} papers...")
        xy = UMAP(**UMAP_2D).fit_transform(config.load_embeddings(fitted, expect=expect))
        layout = pd.DataFrame({'paper_id': fitted.to_numpy(), 'x': xy[:, 0], 'y': xy[:, 1], 'method': 'fit'})

    new = assignments.loc[~assignments['paper_id'].isin(layout['paper_id']), 'paper_id']
    if len(new):
        fit = layout[layout['method'] == 'fit']
        reference = config.load_embeddings(fit['paper_id'], expect=expect)
        vectors = config.load_embeddings(new, expect=expect)
        nearest = np.argsort(-(vectors @ reference.T), axis=1)[:, :config.CONSENSUS_KNN]  # normalized: cosine
        xy = fit[['x', 'y']].to_numpy()[nearest].mean(axis=1)
        layout = pd.concat([layout, pd.DataFrame({'paper_id': new.to_numpy(), 'x': xy[:, 0], 'y': xy[:, 1],
                                                  'method': 'placed'})], ignore_index=True)
        print(f"Placed {len(new)} new papers")
    layout[['x', 'y']] = layout[['x', 'y']].round(4)
    layout.to_csv(path, index=False)


if __name__ == '__main__':
    main()
