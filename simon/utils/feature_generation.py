import geopandas as gpd
from shapely.geometry import Point
import pandas as pd
from scipy.spatial import cKDTree
from utils.copernicus import CopernicusDEM

def generate_land_use_clc(df):
    """
    Incoming dataframe gets an extra feature: land use with clc
    Dataframe needs to have: latitiude and longitude values (here used as coord_lat and coord_lon)
    Also the clc_path needs to be adjusted per machine
    """
    clc_path = 'D:\\Simon\\Documents\\GP\\data\\util_files\\CLC2018\\U2018_CLC2018_V2020_20u1.shp'
    clc = gpd.read_file(clc_path)

    clc_code_to_numerical_label = {
        111: 1,  # Urban
        112: 1,  # Urban
        121: 1,  # Urban
        122: 1,  # Urban
        123: 1,  # Urban
        124: 1,  # Urban
        131: 2,  # Industrial
        132: 2,  # Industrial
        133: 2,  # Industrial
        141: 1,  # Urban
        142: 1,  # Urban
        211: 3,  # Agriculture
        212: 3,  # Agriculture
        213: 3,  # Agriculture
        221: 3,  # Agriculture
        222: 3,  # Agriculture
        223: 3,  # Agriculture
        231: 3,  # Agriculture
        241: 3,  # Agriculture
        242: 3,  # Agriculture
        243: 3,  # Agriculture
        244: 3,  # Agriculture
        311: 4,  # Forest
        312: 4,  # Forest
        313: 4,  # Forest
        321: 5,  # Grassland
        322: 5,  # Grassland
        323: 5,  # Grassland
        324: 5,  # Grassland
        331: 6,  # Water
        332: 6,  # Water
        333: 6,  # Water
        334: 6,  # Water
        335: 6,  # Water
        411: 6,  # Water
        412: 6,  # Water
        421: 6,  # Water
        422: 6,  # Water
        423: 6,  # Water
        511: 6,  # Water
        512: 6,  # Water
        521: 6,  # Water
        522: 6,  # Water
        523: 6,  # Water
        990: 7,  # UNCLASSIFIED LAND SURFACE
        995: 6,  # UNCLASSIFIED WATER BODIES
        999: 8   # NODATA
    }
    numerical_label_to_description = {
        1: 'Urban',
        2: 'Industrial',
        3: 'Agriculture',
        4: 'Forest',
        5: 'Grassland',
        6: 'Water',
        7: 'NODATA',
        8: 'UNCLASSIFIED LAND SURFACE'
}

    # the dataframe coordinates are converted so they can be spatialy merged with clc
    geometry = [Point(lon, lat) for lon, lat in zip(df['coord_lon'], df['coord_lat'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    merged_gdf = gpd.sjoin(gdf, clc, how="left", predicate="within")

    # cleanup and conversion
    merged_gdf.drop(columns=['geometry','index_right', 'OBJECTID', 'Remark', 'Area_Ha', 'ID', 'Shape_Leng', 'Shape_Area'], axis=1, inplace=True)
    merged_gdf['Code_18'].fillna(999, inplace=True)
    merged_gdf['Code_18'] = merged_gdf['Code_18'].astype(int)

    # group landuse by type (and the in same numeric)
    merged_gdf['Numerical_LUSE'] = merged_gdf['Code_18'].map(clc_code_to_numerical_label)
    merged_gdf['LUSE'] = merged_gdf['Numerical_LUSE'].map(numerical_label_to_description)

    merged_gdf.drop(columns=['Code_18'], axis=1, inplace=True)

    return merged_gdf



def generate_land_use_lucas(df):
    """
    Incoming dataframe gets an extra feature: land use with LUCAS
    Dataframe needs to have: latitiude and longitude values (here used as coord_lat and coord_lon)
    Also the lucas_path needs to be adjusted per machine
    """
    lucas_path = 'D:\\Simon\\Documents\\GP\\data\\util_files\\LUCAS\\LUCA_PT_2009.shp'
    lucas_df = gpd.read_file(lucas_path)
    lucas_df.rename(columns={'LAT': 'coord_lat'}, inplace=True)
    lucas_df.rename(columns={'LON': 'coord_lon'}, inplace=True)

    df['geometry'] = gpd.points_from_xy(df['coord_lat'], df['coord_lon'])
    lucas_df['geometry'] = gpd.points_from_xy(lucas_df['coord_lat'], lucas_df['coord_lon'])

    # Create a spatial index (cKDTree) for the points in lucas_df
    tree = cKDTree(lucas_df[['coord_lon', 'coord_lat']])

    # Find the nearest point in lucas_df for each point in df
    df['nearest_idx'] = df.apply(lambda row: tree.query([row['coord_lon'], row['coord_lat']])[1], axis=1)

    merged_df = pd.merge(df, lucas_df[['LUCA_LCOV_', 'LUCA_LUSE_', 'coord_lat', 'coord_lon']], left_on='nearest_idx', right_index=True, suffixes=('_df', '_lucas_df'))

    merged_df.drop(columns=['geometry','nearest_idx'], axis=1, inplace=True)

    return merged_df



def generate_elevation_data():
    """
    Incoming dataframe gets an extra feature: elevation data
    Dataframe needs to have: latitiude and longitude values (here used as coord_lat and coord_lon)
    Also the copernicus_path needs to be adjusted per machine
    """
    copernicus = CopernicusDEM(raster_paths=[
        'D:\\Simon\\Documents\\GP\\data\\util_files\\EU_DEM\\eu_dem_v11_E40N20.TIF', 
        'D:\\Simon\\Documents\\GP\\data\\util_files\\EU_DEM\\eu_dem_v11_E40N30.TIF'])
    df = copernicus.get_elevation(df, lat_col='coord_lat', lon_col='coord_lon')

    df['altitude'] = df['elevation']
    df.drop(columns=['elevation'], inplace=True)

    return df