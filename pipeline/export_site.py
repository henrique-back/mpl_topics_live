"""Data of the public website (site/, served by GitHub Pages): small JSON files of aggregates, no per-paper records
except the anonymous points of the paper map (x, y and topic).

Reads the current taxonomy (topics, AI labels of labels.py, hierarchy), the basins of ner.py, the citations, the 2-D
layout of layout.py and the run records, and writes to site/data/:

  meta.json             update date, numbers of papers, taxonomy version and quality, version history, weekly runs
  taxonomy.json         the topics of each level (label, description, keywords, papers) and the fine topics' parents
  levels/<level>.json   per level: papers per topic and year, topic x basin counts, median citations per year (per
                        topic, per topic x basin cell with >= MIN_CELL_PAPERS rated papers), Gini coefficient per topic
                        and year (cells with >= MIN_GINI_PAPERS rated papers), cumulative citation curves
  basins.json           papers per basin, and per basin and year
  map.json              the paper map: x, y and fine topic of each paper (no ids)
  geo/basins.geojson    the ocean basins, dissolved from the 1-degree grid of ner.py with its MERGE_CLASSES
  geo/land.geojson      the land masses (Natural Earth 1:50m), simplified

The newsletter (site/data/newsletter/) is written by newsletter.py.
"""
import json

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from scipy.spatial import cKDTree
from shapely.geometry import box

import config
import layout
import taxonomy
from analysis import CITES, citation_rates, gini
from ner import BASIN_NAMES, MERGE_CLASSES, read_basin_grid

OUT = config.SITE_DATA_DIR
LEVELS = ['coarse', 'mid', 'fine']
NO_BASIN = 'No basin'
LORENZ_POINTS = 51
LAND_SHAPEFILE = config.MAPS_DIR / 'world map' / 'ne_50m_land.shp'
# Display names of the basins (the classification's own names are abbreviated)
BASIN_DISPLAY = {'Mediteranean': 'Mediterranean Sea', 'Equatorial Pac': 'Equatorial Pacific',
                 'Equatorial Atlant': 'Equatorial Atlantic', 'Equatorial Indian': 'Equatorial Indian',
                 'Arctic': 'Arctic Ocean', 'Antarctic': 'Southern Ocean (Antarctic)', 'Unknown': NO_BASIN}


def write(name, obj):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, separators=(',', ':'), allow_nan=False))
    print(f"  {path.relative_to(config.ROOT)}: {path.stat().st_size / 1024:.0f} kB")


def num(x, digits=3):
    """A JSON number, or None for NaN."""
    return None if pd.isna(x) else round(float(x), digits)


def lorenz(rates):
    """Cumulative share of citations at LORENZ_POINTS evenly spaced shares of papers (least cited first)."""
    rates = np.sort(np.asarray(rates, dtype=float))
    if len(rates) < 2 or rates.sum() == 0:
        return None
    papers = np.arange(len(rates) + 1) / len(rates)
    cites = np.concatenate([[0], np.cumsum(rates)]) / rates.sum()
    return [num(v) for v in np.interp(np.linspace(0, 1, LORENZ_POINTS), papers, cites)]


def load_papers(major):
    data = pd.read_csv(config.ALLDATA, usecols=['DOI', 'Link', 'Year'])
    data['paper_id'] = config.paper_ids(data)
    fine = data['paper_id'].map(taxonomy.load_assignments(major).set_index('paper_id')['topic'])
    if fine.isna().any():
        raise SystemExit(f"{fine.isna().sum()} papers have no topic; run assign.py")
    papers = data[['paper_id', 'Year']].copy()
    papers['fine'] = fine.astype(int).to_numpy()
    for level in ['mid', 'coarse']:
        papers[level] = taxonomy.level_topics(papers['fine'].to_numpy(), level, major)
    basins = pd.read_csv(config.BASIN_LABELS).set_index('paper_id')['Basin']
    papers['basin'] = papers['paper_id'].map(basins).fillna('Unknown').replace(BASIN_DISPLAY)
    papers['rate'] = citation_rates(data)[CITES].to_numpy()
    return papers


def export_taxonomy(papers, major):
    ai = taxonomy.load_labels(major).set_index(['level', 'topic'])
    levels = {}
    for level in LEVELS:
        fallback = taxonomy.labels(major, level=None if level == 'fine' else level)
        sizes = papers.loc[papers[level] != -1, level].value_counts()
        levels[level] = [{'id': int(t), 'n': int(n),
                          'label': ai['label'].get((level, t), fallback[t].split(': ', 1)[1]),
                          'description': ai['description'].get((level, t), ''),
                          'keywords': ai['keywords'].get((level, t), '')}
                         for t, n in sizes.items()]
    parents = papers.drop_duplicates('fine').query('fine != -1').sort_values('fine')[['fine', 'mid', 'coarse']]
    write('taxonomy.json', {'version': taxonomy.current_version(), 'levels': levels,
                            'parents': parents.astype(int).values.tolist(),
                            'outliers': int((papers['fine'] == -1).sum())})


def export_level(papers, level, last_year):
    topics = papers.loc[papers[level] != -1, level].value_counts().index.tolist()
    years = list(range(int(papers['Year'].min()), last_year + 1))
    basins = [b for b in papers['basin'].value_counts().index if b != NO_BASIN] + [NO_BASIN]
    counted = papers[papers[level] != -1]
    over_time = counted[counted['Year'] <= last_year].groupby([level, 'Year']).size().unstack(fill_value=0)
    over_time = over_time.reindex(index=topics, columns=years, fill_value=0)
    per_basin = counted.groupby([level, 'basin']).size().unstack(fill_value=0)
    per_basin = per_basin.reindex(index=topics, columns=basins, fill_value=0)

    rated = counted[counted['rate'].notna()]
    cell = rated.groupby([level, 'basin'])['rate'].agg(['median', 'size'])
    cell_median = cell['median'].where(cell['size'] >= config.MIN_CELL_PAPERS).unstack()
    cell_median = cell_median.reindex(index=topics, columns=basins)
    cell_n = cell['size'].unstack(fill_value=0).reindex(index=topics, columns=basins, fill_value=0)

    gini_rows = []
    for (t, year), rates in rated.groupby([level, 'Year'])['rate']:
        if len(rates) >= config.MIN_GINI_PAPERS:
            gini_rows.append([int(t), int(year), len(rates), num(gini(rates))])
    by_topic = rated.groupby(level)['rate']
    write(f'levels/{level}.json', {
        'topics': [int(t) for t in topics], 'years': years, 'basins': basins,
        'over_time': over_time.values.tolist(),
        'basin_counts': per_basin.values.tolist(),
        'median_rate': [num(by_topic.median().get(t)) for t in topics],
        'rated': [int(by_topic.size().get(t, 0)) for t in topics],
        'gini_topic': [num(gini(rated.loc[rated[level] == t, 'rate'])) for t in topics],
        'cell_median_rate': [[num(v) for v in row] for row in cell_median.values],
        'cell_rated': cell_n.values.tolist(),
        'gini': gini_rows,
        'lorenz': [lorenz(rated.loc[rated[level] == t, 'rate']) for t in topics],
        'lorenz_all': lorenz(rated['rate'])})


def export_basins(papers, last_year):
    years = list(range(int(papers['Year'].min()), last_year + 1))
    counts = papers['basin'].value_counts()
    by_year = (papers[papers['Year'] <= last_year].groupby(['basin', 'Year']).size().unstack(fill_value=0)
               .reindex(columns=years, fill_value=0))
    write('basins.json', {'years': years,
                          'basins': [{'name': b, 'n': int(n), 'by_year': by_year.loc[b].tolist()}
                                     for b, n in counts.items()]})


def export_map(papers, major):
    xy = layout.load_layout(major).set_index('paper_id').reindex(papers['paper_id'])
    keep = xy['x'].notna().to_numpy()
    if not keep.all():
        print(f"  {(~keep).sum()} papers have no map position; run layout.py")
    write('map.json', {'x': [round(v, 2) for v in xy['x'][keep]], 'y': [round(v, 2) for v in xy['y'][keep]],
                       'topic': papers.loc[keep, 'fine'].astype(int).tolist()})


def export_geo():
    """The ocean basins (merged classes, named as the papers' basins) and the land, as GeoJSON."""
    grid = read_basin_grid()
    # The grid misses the cells along the equator and the 0-degree meridian: fill every cell of the full 1-degree grid
    # with the basin of the nearest grid value (as basin_map.fill_grid_gaps)
    lat, lon = np.meshgrid(np.arange(-89.5, 90), np.arange(-179.5, 180), indexing='ij')
    full = pd.DataFrame({'Latitude': lat.ravel(), 'Longitude': lon.ravel()})
    _, nearest = cKDTree(grid[['Latitude', 'Longitude']].to_numpy()).query(full.to_numpy())
    full['range_area'] = grid['range_area'].to_numpy()[nearest]
    print(f"  basin grid: {len(full) - len(grid)} missing cells filled")
    grid = full
    merge = {code: group[0] for group in MERGE_CLASSES for code in group}
    names = pd.read_csv(BASIN_NAMES, names=['code', 'basin']).astype({'code': float}).set_index('code')['basin']
    grid = grid[grid['range_area'] != 1].copy()  # 'Climatology Range', left out of the map as in basin_map.py
    grid['basin'] = grid['range_area'].replace(merge).astype(float).map(names).replace(BASIN_DISPLAY)
    # Each grid value is the centre of its 1-degree cell
    cells = gpd.GeoDataFrame(grid[['basin']], geometry=[box(lon - .5, lat - .5, lon + .5, lat + .5) for lat, lon in
                                                        zip(grid['Latitude'], grid['Longitude'])])
    basins = cells.dissolve('basin').reset_index()
    basins['geometry'] = shapely.set_precision(basins.simplify(0.2).values, 0.01)
    basins = basins.rename(columns={'basin': 'name'})
    write('geo/basins.geojson', json.loads(basins.to_json(drop_id=True, to_wgs84=False)))
    land = gpd.read_file(LAND_SHAPEFILE)[['geometry']]
    land = gpd.GeoDataFrame({'name': ['Land']}, geometry=[shapely.set_precision(land.union_all().simplify(0.15), 0.01)])
    write('geo/land.geojson', json.loads(land.to_json(drop_id=True)))


def export_meta(papers, major, last_year):
    params = taxonomy.load_params(major)
    runs = pd.read_csv(taxonomy.RUNS) if taxonomy.RUNS.exists() else pd.DataFrame()
    added = pd.read_csv(config.ADDED) if config.ADDED.exists() else pd.DataFrame(columns=['added_on'])
    week = added['added_on'].max() if len(added) else None
    versions = taxonomy.versions()
    crosswalk_path = config.TAXONOMY_DIR / major / 'crosswalk.csv'
    crosswalk = None
    if crosswalk_path.exists():
        cw = pd.read_csv(crosswalk_path)
        matched = cw[cw['jaccard'] >= 0.1] if 'jaccard' in cw else cw.iloc[:0]
        previous = versions['version'].map(taxonomy.major).drop_duplicates()
        previous = previous[previous != major]
        crosswalk = {'previous': previous.iloc[-1] if len(previous) else None,
                     'matched': int(matched['new_topic'].nunique()), 'old_topics': int(cw['old_topic'].nunique()),
                     'new_topics': int(papers.loc[papers['fine'] != -1, 'fine'].nunique())}
    write('meta.json', {
        'updated': str(config.reference_date()), 'n_papers': len(papers),
        'week': week, 'n_week': int((added['added_on'] == week).sum()) if week else 0,
        'first_year': int(papers['Year'].min()), 'last_full_year': last_year,
        'n_located': int((papers['basin'] != NO_BASIN).sum()),
        'n_rated': int(papers['rate'].notna().sum()),
        'outlier_share': num((papers['fine'] == -1).mean()),
        'n_topics': {level: int(papers.loc[papers[level] != -1, level].nunique()) for level in LEVELS},
        'version': taxonomy.current_version(), 'labels_model': config.CLAUDE_MODEL,
        'quality': {'runs': params.get('consensus', {}).get('runs'),
                    'split_half': params.get('consensus', {}).get('split_half'),
                    'fit': params.get('fit'), 'encoder': config.EMBEDDING_MODEL},
        'thresholds': {'min_citation_age': config.MIN_CITATION_AGE, 'min_gini_papers': config.MIN_GINI_PAPERS,
                       'min_cell_papers': config.MIN_CELL_PAPERS, 'levels': config.TOPIC_LEVELS},
        'versions': versions.astype(object).where(versions.notna(), None).to_dict('records'),
        'crosswalk': crosswalk,
        'runs': runs.tail(26).astype(object).where(runs.tail(26).notna(), None).to_dict('records'),
        'rebaseline_due': bool(len(runs)) and str(runs['rebaseline_due'].iloc[-1]).lower() == 'true'})


def main():
    major = taxonomy.major(taxonomy.current_version())
    last_year = config.last_full_year()
    papers = load_papers(major)
    print(f"Exporting the site data of {len(papers)} papers ({taxonomy.current_version()})")
    export_meta(papers, major, last_year)
    export_taxonomy(papers, major)
    for level in LEVELS:
        export_level(papers, level, last_year)
    export_basins(papers, last_year)
    export_map(papers, major)
    export_geo()


if __name__ == '__main__':
    main()
