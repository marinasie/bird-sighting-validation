import csv
import numpy as np
import pandas as pd
import geopandas as gpd


def map_atlas_code(atlas_code):
    try:
        if np.isnan(atlas_code):
            return 0
        atlas_code = int(atlas_code)
        if 1 <= atlas_code <=3 or atlas_code==30:
            return 1
        elif 4 <= atlas_code <=10 or atlas_code==40:
            return 2
        elif 11 <= atlas_code <=20 or atlas_code==50:
            return 3
        return 0
    except (ValueError, TypeError):
        if atlas_code.startswith('A'):
            return 1
        elif atlas_code.startswith('B'):
            return 2
        elif atlas_code.startswith('C'):
            return 3
        return 0


def condense_atlas_codes(codes: pd.Series) -> pd.Series:
    """
    Converts all atlas codes to the following code groups:
    - no breeding (0)
    - possible breeding (1)
    - likely breeding (2)
    - confirmed breeding (3)
    """
    return codes.apply(map_atlas_code)


def get_delimiter(file_path:str, bytes = 4096):
    sniffer = csv.Sniffer()
    data = open(file_path, 'r').read(bytes)
    delimiter = sniffer.sniff(data).delimiter
    return delimiter


def standardize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Changes the column dtypes to our standardized column dtypes.
    """
    df.id_species = df.id_species.astype('Int64')
    df.total_count = df.total_count.astype('Int64')
    if df.coord_lat.dtype != np.float64:
        df.coord_lat = pd.to_numeric(df.coord_lat.str.replace(',', '.'), errors='coerce')
        df.coord_lon = pd.to_numeric(df.coord_lon.str.replace(',', '.'), errors='coerce')
    if 'id_observer' in df.columns:
        df.id_observer = df.id_observer.astype('Int64')
    return df


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Changes the column names to our standardized column names.
    """
    df.columns = df.columns.str.lower()
    df.rename({'atlas_code_ch': 'atlas_code'}, axis='columns', inplace=True)
    df.rename({'beobachter':'id_observer', 'altas_code': 'atlas_code'}, axis='columns', inplace=True)
    return df


def standardize_date_format(df:pd.DataFrame, format:str) -> pd.DataFrame:
    """
    Changes the date format to our standardized format yyyy-mm-dd.
    """
    df.date = pd.to_datetime(df.date, format=format)
    df.date = df.date.dt.strftime('%Y-%m-%d')
    return df
    

def standardize_precisions(df:pd.DataFrame) -> pd.DataFrame:
    """
    Changes precision names to standardized precisions and drops other (minority) precision names.
    """
    precisions = {'Exakte Lokalisierung': 'precise',
                  'Kilometerquadrat': 'square',
                  'Ort': 'place',
                  'Halbminutenfeld': 'square',
                  'exakte Lokalisierung (Linie)': 'precise',
                  'exakte Lokalisierung (Polygon)': 'precise',
                  'exakte Lokalisierung (Punkt)': 'precise',
                  'Polygon': 'square',
                  'Punkt': 'precise',
                  'Transekt': 'square'}
    df.precision = df.precision.map(precisions).fillna(df.precision)
    df = df[df.precision.isin(['precise', 'square', 'place'])]
    return df


def standardize_id_species(df:pd.DataFrame, path_to_lookup_table:str)  -> pd.DataFrame:
    """
    Changes german species id to standardized species id's given by 'translation_species_id_de_vs_ch.csv'
    """
    translator = pd.read_csv(path_to_lookup_table, delimiter=';')
    translator_dict = dict(zip(translator.id_species_dbird, translator.id_species_ornitho))
    df.loc[:, 'id_species'] = df.id_species.map(translator_dict).fillna(df.id_species)
    return df


def standardize_name_species(df:pd.DataFrame, path_to_lookup_table:str)  -> pd.DataFrame:
    """
    Changes swiss species names to standardized names given by 'translation_species_names_de_vs_ch.csv'
    """
    translator = pd.read_csv(path_to_lookup_table)
    species_map = dict(zip(translator.id_species, translator.name_species))
    df.loc[:, 'name_species'] = df.id_species.map(species_map).fillna(df.name_species)
    return df


def assign_eea_grids(df:pd.DataFrame, eea_shapefile_path:str)  -> pd.DataFrame:
    """
    Assigns an eea grid id to every data point based on the provided shapefile.
    """
    # Assign eea grids
    eea_grid = gpd.read_file(eea_shapefile_path)
    eea_grid = eea_grid.to_crs('EPSG:4326')

    germany_switzerland_bbox = eea_grid.cx[5.210942:15.669926, 45.614516:55.379499]
    eea_grid_filtered = eea_grid[eea_grid.intersects(germany_switzerland_bbox.unary_union)]
    eea_grid_filtered.reset_index(drop=True, inplace=True)

    geometry = gpd.points_from_xy(df['coord_lon'], df['coord_lat'])
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')

    df = gpd.sjoin(gdf, eea_grid_filtered, how='left', predicate='within')
    df.rename(columns={'cellcode': 'eea_grid_id'}, inplace=True)
    df = df.drop(columns=['index_right', 'geometry', 'noforigin', 'eoforigin', 'gid'])
    return df


def standardize_data(df:pd.DataFrame, path_translator_species_names:str, adjust_ids=False, path_translator_species_ids=None, date_format='%Y-%m-%d') -> pd.DataFrame:
    """
    Modifies the dataframes from ornitho to our standard data pattern
    """
    if 'estimation_code' in df.columns:
        df.drop(columns=['estimation_code'], inplace=True)
    df = standardize_column_names(df)
    df = standardize_dtypes(df)
    df = standardize_date_format(df, format=date_format)
    df = standardize_precisions(df)
    df = standardize_name_species(df, path_to_lookup_table=path_translator_species_names)
    if adjust_ids:
        df = standardize_id_species(df, path_translator_species_ids)
    return df


def extract_digits_from_str(input_string:str):
    result_string = ''.join(filter(str.isdigit, input_string))
    return int(result_string)


def numerize_eea_grids(grid_ids: pd.Series) -> pd.Series:
    """
    Converts the EEA grid ids to numeric values.
    """
    return grid_ids.apply(extract_digits_from_str)