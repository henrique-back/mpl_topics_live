"""Step 4: assign each paper to an ocean basin from the locations named in its abstract.

Stages (run in order; `--from` starts at a later one):
  extract   spaCy NER: GPE/LOC entities of the abstracts not processed yet (papers of alldata.csv whose paper_id
            is not in NER/data_loc.csv) -> NER/data_loc.csv, NER/all_locations.csv  (slow)
  geocode   GeoNames API: coordinates of the locations not looked up yet (neither in fuzzy_location_data.csv nor in
            fuzzy_error_locations.csv) -> NER/fuzzy_location_data.csv, NER/fuzzy_error_locations.csv  (slow, external
            API). Progress is saved as it goes; if the API credit limit is reached, re-run later to continue. To look
            up a failed location again, remove it from fuzzy_error_locations.csv.
  manual    offline: manual coordinates (ERROR_DICT), nearest ocean-basin grid cell of each location, and the
            majority-vote basin of each paper -> NER/merged_fuzzy_classes_labels.csv, NER/maps/range_area_counts.csv
"""
import argparse
import time

import numpy as np
import pandas as pd
import requests
from sklearn.neighbors import BallTree

import config

DATA_LOC = config.NER_DIR / 'data_loc.csv'
ALL_LOCATIONS = config.NER_DIR / 'all_locations.csv'
LOCATION_DATA = config.NER_DIR / 'fuzzy_location_data.csv'
ERROR_LOCATIONS = config.NER_DIR / 'fuzzy_error_locations.csv'
BASIN_GRID = config.NER_DIR / 'range_area.csv'
BASIN_NAMES = config.NER_DIR / 'range_basin_list.csv'

# Manual coordinates for locations that GeoNames could not find or placed wrongly
ERROR_DICT = {"Arafura-Timor Seas": [-9.5, 134.5], "Punnakayal": [8.632222, 78.111950],
              "Aguincheira": [40.835621, -8.380748], "Rhine-Ruhr": [51.451770, 7.018720],
              "Qinghai-Tibet Plateau": [35.751230, 94.988449], "Fukutoku-Okanoba": [34.004161, 134.563027],
              "Odaw": [5.548512, -0.224942], "Chellanam": [9.807211, 76.277420], "South Malang": [-7.979470, 112.634222],
              "Cabo-Delgado": [-12.750000, 39.500000], "Rhine-Meuse": [51.730431, 4.715880],
              "Ho-Chi-Minh city": [10.668271, 106.482262], "Solimões": [-2.601662, -55.143238],
              "Santos Bay": [-23.959779, -46.344984], "Perú": [-11.185900, -76.841576],
              "North Maluku Province": [1.348434, 127.668530],
              "Patok-Fushëkuqe-Ishëm": [41.628820, 19.619820], "High-Arctic": [79.138640, -3.238970],
              "Sanbanze": [35.672459, 139.967468],
              "Apulo-Lucanian": [40.371390, 16.811220], "Lohabarrack": [11.603486, 92.609825],
              "Alto-Tiete": [-23.060740, -47.723861],
              "Kinmen-Xiamen": [24.479720, 118.081900], "Kodiyaghat": [-2.780561, -78.854599]}

# Coastal and open-ocean codes of the same basin are merged into the first code of each group
MERGE_CLASSES = [[2, 3, 1], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15], [16, 17], [18, 19]]


# ---------------------------------------------------------------------------------------------
# extract
# ---------------------------------------------------------------------------------------------

def extract_locations():
    """Extract the GPE and LOC entities of the abstracts that have no row in DATA_LOC yet, with en_core_web_trf.

    Only new papers (by paper_id) are processed, and existing rows are kept: re-running the extraction in another
    spaCy / model environment changes the locations of some papers (see KNOWN_ISSUES.md). Delete DATA_LOC to start
    over.
    """
    data = pd.read_csv(config.ALLDATA)
    data['paper_id'] = config.paper_ids(data)
    # Only empty cells are missing values, so that a location named e.g. "NA" is written back unchanged
    data_loc = (pd.read_csv(DATA_LOC, dtype=str, keep_default_na=False, na_values=[''])
                if DATA_LOC.exists() else pd.DataFrame(columns=['paper_id']))
    new_papers = data[~data['paper_id'].isin(data_loc['paper_id'])]
    new_abstracts = new_papers['Abstract']
    print(f"Extracting locations from {len(new_abstracts)} new abstracts ({len(data_loc)} already done)")

    if not len(new_abstracts):
        return

    import spacy
    from spacy.lang.en.stop_words import STOP_WORDS

    # Only the named-entity recognizer is needed
    nlp = spacy.load('en_core_web_trf', disable=["tagger", "parser", "attribute_ruler", "lemmatizer"])

    # Locations of each abstract, in order of first mention and without duplicates
    locations = []
    for abstract in new_abstracts:
        doc_locations = {}
        for ent in nlp(abstract).ents:
            if ent.label_ in ('GPE', 'LOC'):
                filtered_location = " ".join(word for word in ent.text.split() if word.lower() not in STOP_WORDS)
                if filtered_location:
                    doc_locations[filtered_location] = None
        locations.append(list(doc_locations))

    # Locations of each paper: one row per paper, keyed on paper_id
    new_loc = pd.DataFrame(locations)
    new_loc.columns = new_loc.columns.astype(str)
    new_loc.insert(0, 'paper_id', new_papers['paper_id'].values)
    data_loc = pd.concat([data_loc, new_loc])
    data_loc.to_csv(DATA_LOC, index=False)

    # Append the locations not seen before to the list of unique locations, in order of first mention
    exists = ALL_LOCATIONS.exists()
    known = set(pd.read_csv(ALL_LOCATIONS, keep_default_na=False)['Location']) if exists else set()
    new_locations = [loc for loc in pd.unique(new_loc.drop(columns='paper_id').to_numpy().ravel())
                     if isinstance(loc, str) and loc not in known]
    pd.DataFrame({'Location': new_locations}).to_csv(ALL_LOCATIONS, mode='a', header=not exists, index=False)
    print(f"{len(new_locations)} new unique locations ({len(known) + len(new_locations)} in total)")


# ---------------------------------------------------------------------------------------------
# geocode
# ---------------------------------------------------------------------------------------------

class GeoNamesError(Exception):
    """The API refused the request (e.g. credit limit exceeded, invalid username)."""


def search_location(location_name, username, fuzzy=False):
    params = {"name": location_name, "maxRows": 50, "username": username}
    if fuzzy:
        params["fuzzy"] = 0.8
    data = requests.get("http://api.geonames.org/searchJSON", params=params, timeout=60).json()
    # Errors come back as a 'status' message instead of results. They must not be mistaken for "not found",
    # since locations that were not found are not looked up again.
    if "status" in data:
        raise GeoNamesError(data["status"].get("message", data["status"]))
    return data


def geocode(location_name, username):
    """Exact search, then fuzzy search, then fuzzy search on each word. Returns None if nothing is found."""
    data = search_location(location_name, username)
    if data.get("geonames"):
        match, kind = data["geonames"][0], "exact"
    else:
        data = search_location(location_name, username, fuzzy=True)
        if data.get("geonames"):
            match, kind = data["geonames"][0], "fuzzy"
        else:
            for word in location_name.split():
                data = search_location(word, username, fuzzy=True)
                if data.get("geonames"):
                    match = data["geonames"][0]
                    kind = match.get("name", "Unknown")
                    break
            else:
                return None
    return {
        "location": kind,
        "feature_class": match.get("fcl", "Unknown"),
        "latitude": match.get("lat", "Unknown"),
        "longitude": match.get("lng", "Unknown"),
    }


def geocode_locations(delay=2, save_every=25):
    """Geocode the locations of ALL_LOCATIONS that were not looked up before, waiting `delay` seconds between
    locations (API rate limits).

    Found locations are appended to LOCATION_DATA (with no basin yet, so that assign_basins computes only theirs) and
    the ones GeoNames does not know to ERROR_LOCATIONS. Locations hit by a network or API error are left out of both
    files, so the next run tries them again.
    """
    username = config.read_secrets()['geonames']['username']

    location_df = (pd.read_csv(LOCATION_DATA, float_precision='round_trip') if LOCATION_DATA.exists()
                   else pd.DataFrame(columns=['Location']))
    error_locations = pd.read_csv(ERROR_LOCATIONS)['Error Locations'].tolist() if ERROR_LOCATIONS.exists() else []
    known = set(location_df['Location']) | set(error_locations) | set(ERROR_DICT)
    # Names read as missing values (e.g. "NA") are skipped, as in classify_papers
    todo = [loc for loc in pd.read_csv(ALL_LOCATIONS)['Location'].dropna() if loc not in known]
    print(f"Geocoding {len(todo)} new locations ({len(known)} already done)")
    if not todo:
        return

    found = {}

    def save():
        new_df = pd.DataFrame.from_dict(found, orient='index').rename_axis('Location').reset_index()
        df = pd.concat([location_df, new_df], ignore_index=True)
        if 'Closest Ocean Basin' in df.columns:
            df['Closest Ocean Basin'] = df['Closest Ocean Basin'].astype('Int64')  # keep integer codes next to NaN
        df.to_csv(LOCATION_DATA, index=False)
        pd.DataFrame({'Error Locations': error_locations}).to_csv(ERROR_LOCATIONS, index=False)

    try:
        for i, location_name in enumerate(todo, 1):
            try:
                result = geocode(location_name, username)
            except requests.RequestException as e:
                print(f"Skipped '{location_name}' ({e}); it will be tried again on the next run")
                result = False
            if result is None:
                print(f"Location '{location_name}' not found in GeoNames database")
                error_locations.append(location_name)
            elif result:
                found[location_name] = result
            if i % save_every == 0:
                save()
            time.sleep(delay)
    finally:
        # Also on an API error or an interruption, so that a re-run continues where this one stopped
        save()
        print(f"{len(found)} locations found, {len(error_locations)} not found in total")


# ---------------------------------------------------------------------------------------------
# manual (offline)
# ---------------------------------------------------------------------------------------------

def apply_manual_coordinates():
    """Set the coordinates in ERROR_DICT and clear those locations' basins, so that they are recomputed."""
    location_df = pd.read_csv(LOCATION_DATA, float_precision='round_trip').set_index('Location')
    for name, (lat, lon) in ERROR_DICT.items():
        location_df.loc[name, ['location', 'feature_class', 'latitude', 'longitude']] = ['manual', 'manual', lat, lon]
        if 'Closest Ocean Basin' in location_df.columns:
            location_df.loc[name, ['Closest Ocean Basin', 'Distance to Nearest Ocean']] = np.nan
    location_df.reset_index().to_csv(LOCATION_DATA, index=False)

    # Remove the manually located places from the error list
    error_locations = pd.read_csv(ERROR_LOCATIONS)['Error Locations']
    error_locations = error_locations[~error_locations.isin(list(ERROR_DICT))].drop_duplicates()
    error_locations.to_csv(ERROR_LOCATIONS, index=False)
    print(f"{len(ERROR_DICT)} locations set manually; {len(error_locations)} locations remain unresolved")


def read_basin_grid():
    """The 1-degree ocean-basin grid, without the cells with invalid coordinates."""
    range_area = pd.read_csv(BASIN_GRID)
    return range_area[range_area['Latitude'].between(-90, 90) & range_area['Longitude'].between(-180, 180)].copy()


def assign_basins():
    """Assign each location to the closest cell of the 1-degree ocean-basin grid (great-circle distance).

    Only locations without a basin are (re)computed: new locations and manually corrected ones. Locations
    exactly equidistant to two grid cells are tie-broken differently across scikit-learn versions, so existing
    assignments are kept to make the classification reproducible.
    """
    location_df = pd.read_csv(LOCATION_DATA, float_precision='round_trip')
    range_area = read_basin_grid()

    if 'Closest Ocean Basin' not in location_df.columns:
        location_df['Closest Ocean Basin'] = np.nan
        location_df['Distance to Nearest Ocean'] = np.nan
    todo = location_df['Closest Ocean Basin'].isna()

    if todo.any():
        tree = BallTree(np.radians(range_area[['Latitude', 'Longitude']].values), metric='haversine')
        coords = np.radians(location_df.loc[todo, ['latitude', 'longitude']].values.astype(float))
        distances, indices = tree.query(coords, k=1)
        location_df.loc[todo, 'Closest Ocean Basin'] = range_area['range_area'].values[indices[:, 0]]
        location_df.loc[todo, 'Distance to Nearest Ocean'] = distances[:, 0] * 6371  # radians -> km
    location_df['Closest Ocean Basin'] = location_df['Closest Ocean Basin'].astype(int)

    location_df.to_csv(LOCATION_DATA, index=False)
    print(f"Assigned {todo.sum()} locations to a basin")

    # Number of locations by distance to the nearest ocean-basin grid cell
    bins = [0, 50, 100, 500, 1000, np.inf]
    labels = ['< 50 km', '50-100 km', '100-500 km', '500-1000 km', '>= 1000 km']
    print(pd.cut(location_df['Distance to Nearest Ocean'], bins=bins, labels=labels, right=False)
          .value_counts().sort_index().to_string())


def classify_papers():
    """Classify each paper by the most frequent basin of its locations, merging coastal and open-ocean codes."""
    data_loc = pd.read_csv(DATA_LOC).set_index('paper_id')
    location_df = pd.read_csv(LOCATION_DATA, float_precision='round_trip')

    # Replace each location by the code of its closest basin. Locations that could not be geocoded
    # become NaN, so that they do not take part in the vote below.
    location_dict = location_df.set_index('Location')['Closest Ocean Basin'].to_dict()
    labels_df = data_loc.map(lambda val: location_dict.get(val, np.nan) if pd.notna(val) else val)

    # Most frequent basin of each paper (None if none of its locations was found)
    classes = [row.dropna().value_counts().idxmax() if row.notna().any() else None
               for _, row in labels_df.iterrows()]
    classes_df = pd.DataFrame({'paper_id': data_loc.index, 'Most Recurrent Value': pd.array(classes, dtype='Int64')})

    range_area = read_basin_grid()
    range_area['range_area'] = range_area['range_area'].astype(float)

    # Remove the differentiation between coastal and oceanic basins (i.e. coastal north atlantic and north atlantic)
    merge_mapping = {code: group[0] for group in MERGE_CLASSES for code in group}
    classes_df['Merged Class'] = classes_df['Most Recurrent Value'].replace(merge_mapping)
    basin_counts = classes_df['Merged Class'].value_counts()

    # Number of papers of each grid cell's (merged) basin, for the map
    range_area['Merged Class'] = range_area['range_area'].replace(merge_mapping)
    range_area['basin_count'] = range_area['Merged Class'].map(basin_counts)

    # Include the ocean label besides the code
    range_basin_list = pd.read_csv(BASIN_NAMES, names=['range_area', 'basin'])
    range_basin_list['range_area'] = range_basin_list['range_area'].astype(float).round(1)
    range_area = range_area.merge(range_basin_list, on='range_area', how='left')
    range_area = range_area.drop(['range_area', 'Merged Class'], axis=1)

    label_mapping = dict(zip(range_basin_list['range_area'], range_basin_list['basin']))
    classes_df['Basin'] = classes_df['Merged Class'].map(label_mapping)

    range_area.to_csv(config.BASIN_GRID_COUNTS, index=False)
    classes_df.to_csv(config.BASIN_LABELS, index=False)
    print(classes_df['Basin'].value_counts(dropna=False).to_string())


STAGES = {
    'extract': [extract_locations],
    'geocode': [geocode_locations],
    'manual': [apply_manual_coordinates, assign_basins, classify_papers],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--from', dest='start', choices=list(STAGES), default='extract',
                        help="stage to start from (default: extract, i.e. run everything)")
    start = parser.parse_args().start
    names = list(STAGES)
    for name in names[names.index(start):]:
        for step in STAGES[name]:
            step()


if __name__ == '__main__':
    main()
