"""Step 5: world choropleth of the number of papers per ocean basin -> graphs/map.svg, graphs/map.png."""
import geopandas as gpd
import matplotlib

matplotlib.use('Agg')  # write files only, no window
matplotlib.rcParams['svg.hashsalt'] = 'mpl_topics'  # stable SVG element ids, so the file only changes with the map

import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from shapely.geometry import Point, Polygon
from sklearn.neighbors import NearestNeighbors

import config

LAND_SHAPEFILE = config.MAPS_DIR / 'world map' / 'ne_50m_land.shp'


def fill_grid_gaps(range_area):
    """Fill the gaps along the 0-degree meridian and the equator with the basin of the nearest grid point."""
    # Points with longitude 0.5 and -0.5 for all latitudes, and latitude 0.5 and -0.5 for all longitudes
    new_points = pd.concat([
        pd.DataFrame({'Latitude': [0.5] * 361, 'Longitude': range(-180, 181)}),
        pd.DataFrame({'Latitude': [-0.5] * 361, 'Longitude': range(-180, 181)}),
        pd.DataFrame({'Latitude': range(-90, 91), 'Longitude': [0.5] * 181}),
        pd.DataFrame({'Latitude': range(-90, 91), 'Longitude': [-0.5] * 181}),
    ])
    grid_points_df = gpd.GeoDataFrame(geometry=[Point(lon, lat) for lat, lon in
                                                zip(new_points['Latitude'], new_points['Longitude'])])

    # Nearest existing grid point of each new point
    existing_points_array = range_area[['Longitude', 'Latitude']].to_numpy(dtype=float)
    knn = NearestNeighbors(n_neighbors=5).fit(existing_points_array)
    grid_points_array = np.array([[p.x, p.y] for p in grid_points_df.geometry])
    _, indices = knn.kneighbors(grid_points_array)

    grid_points_df['basin'] = range_area.loc[indices[:, 0], 'basin'].values
    grid_points_df['basin_count'] = range_area.loc[indices[:, 0], 'basin_count'].values
    grid_points_df['Latitude'] = grid_points_df.geometry.y
    grid_points_df['Longitude'] = grid_points_df.geometry.x
    return pd.concat([range_area, grid_points_df[['Latitude', 'Longitude', 'basin', 'basin_count']]],
                     ignore_index=True)


def basin_polygons(range_area_filled):
    """One 1-degree square per grid point, with its basin and number of papers."""
    ocean_polygons = []
    for _, row in range_area_filled.iterrows():
        lat, lon = row['Latitude'], row['Longitude']
        square = Polygon([(lon, lat), (lon, lat + 1), (lon + 1, lat + 1), (lon + 1, lat), (lon, lat)])
        ocean_polygons.append((row['basin_count'], row['basin'], square))
    ocean_df = gpd.GeoDataFrame(ocean_polygons, columns=['basin_count', 'basin', 'geometry'], geometry='geometry')
    return ocean_df[ocean_df['basin'] != 'Climatology Range']


def plot_map(ocean_df, continents):
    fig, ax = plt.subplots(figsize=(20, 20))
    continents.plot(ax=ax, color='lightgray', edgecolor='black', zorder=2)

    # Discrete colormap with one step per distinct number of papers
    cmap = cm.inferno
    unique_values = np.unique(ocean_df['basin_count'])
    bounds = np.concatenate(([0], unique_values))
    norm = colors.BoundaryNorm(bounds, cmap.N, extend='max')

    ocean_df.plot(ax=ax, column='basin_count', categorical=False, legend=False,
                  cmap=cmap, norm=norm, linewidth=0.8, edgecolor='none', zorder=1)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('bottom', size='5%', pad=0.3)
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),
                        cax=cax, orientation='horizontal', spacing='proportional', extendrect=True, extendfrac='auto')

    # Label the papers-per-basin values, skipping values too close to the next larger label to be legible
    ticks = []
    for value in sorted(unique_values[~np.isnan(unique_values)].astype(int), reverse=True):
        if not ticks or ticks[-1] - value >= 50:
            ticks.append(value)
    ticks = sorted(ticks)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(ticks, fontsize=14)
    cbar.ax.xaxis.set_tick_params(pad=10)

    ax.set_title("")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    return fig


def main():
    # Number of papers per basin on the 1-degree grid (written by ner.py)
    range_area = pd.read_csv(config.BASIN_GRID_COUNTS)
    continents = gpd.read_file(LAND_SHAPEFILE)

    ocean_df = basin_polygons(fill_grid_gaps(range_area))
    fig = plot_map(ocean_df, continents)

    # No date in the metadata, so the files only change when the map does
    fig.savefig(config.GRAPHS_DIR / 'map.svg', format='svg', bbox_inches='tight', metadata={'Date': None})
    fig.savefig(config.GRAPHS_DIR / 'map.png', format='png', dpi=300, bbox_inches='tight')

    # Papers per basin
    print(ocean_df.drop_duplicates(subset='basin')[['basin', 'basin_count']].to_string(index=False))


if __name__ == '__main__':
    main()
