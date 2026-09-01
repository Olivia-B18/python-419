"""
Olivia Buchanan
Homework 0: Setting up Data Set

ABOUT:
This script imports all csv files from the raw_data/
directory into DataFrames, cleans the data, and
then saves the processed data as Parquet files to be
loaded by load.py 

DATA:
raw_data/*
"""

from pathlib import Path
import pandas as pd
import numpy as np

#####################################
# CONSTANTS
#####################################

# This schema helps with checking and casting columns
# to correct data types. Refer to sanitize_cols().
# 
# NOTE: since whitespace and quotes are normalized
# during sanitization, sentinels can be matched exactly.

SCHEMA = {
    "string": {
        "check_fxn": pd.api.types.is_string_dtype,
        "cast_type": "string",
        "target_cols": [
            "track_id",
            "artists",
            "song",
            "album"
        ],
        "sentinels": [
            "",
            " ",
            "None",
            "none",
            "null",
            "NULL",
            "N/A",
            "n/a",
            "NA",
            "UNKNOWN",
            "unknown"
        ],
    },
    "category": {
        "check_fxn": lambda s: isinstance(s.dtype, pd.CategoricalDtype),
        "cast_type": "category",
        "target_cols": [
            "time_signature",
            "mode",
            "key",
            "genre"
        ],
        "sentinels": [
            "",
            "None",
            "null",
            "UNKNOWN",
            "N/A",
            "Unassigned"
        ],
    },
    "float": {
        "check_fxn": pd.api.types.is_float_dtype,
        "cast_type": "float64",
        "target_cols": [
            "loudness",
            "speechiness",
            "acousticness",
            "instrumentalness",
            "liveness",
            "valence",
            "tempo",
            "duration_ms",
            "danceability",
            "energy",
        ],
        "sentinels": [
            -1.0,
            -999.0,
            -9999.0,
            999.0,
            9999.0
        ],
    },
    "int": {
        "check_fxn": pd.api.types.is_integer_dtype,
        "cast_type": "Int64",
        "target_cols": [
            "popularity",
            "year"
        ],
        "sentinels": [
            -1,
            -999,
            -9999,
            999,
            9999
        ],
    },
    "bool": {
        "check_fxn": pd.api.types.is_bool_dtype,
        "cast_type": "boolean",
        # astype(bool) is truthiness-based, so it would turn the
        # string "False" into True. Map known spellings instead and
        # let anything unrecognized become NA so the row is dropped.
        "cast_fxn": lambda s: to_bool(s),
        "target_cols": ["explicit"],
        "sentinels": [
            "",
            "None",
            "null",
            "UNKNOWN",
            "N/A",
            -1
        ],
    },
}

# Required columns both raw DataFrames should have
REQ_COMMON_COLS= {
        "track_id",
        "artists",
        "song",
        "popularity",
        "duration_ms",
        "danceability",
        "energy",
        "key",
        "loudness",
        "mode",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
        "time_signature",
        }

# Map to set invalid values to bools in a column
BOOL_MAP = {
    True: True, "True": True, "true": True, "TRUE": True,
    "T": True, "t": True, "1": True, 1: True,
    False: False, "False": False, "false": False, "FALSE": False,
    "F": False, "f": False, "0": False, 0: False,
}

#####################################
# FUNCTIONS
#####################################

def dir_exists(input_dir: Path):
    """
    Checks that the input directory exists.

    Parameters:
        input_dir (Path): Input directory.

    Raises:
        FileNotFoundError: If the directory does not exist.
    """
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {input_dir}")

def to_bool(input_series: pd.Series) -> pd.Series:
    """
    Converts a Series to nullable boolean via explicit mapping.

    Parameters:
        input_series (pd.Series): Input Series.

    Returns:
        pd.Series: 'boolean' dtype Series. Values not recognized
        as a spelling of True/False become pd.NA so that the row
        may be dropped later.
    """
    return input_series.map(BOOL_MAP).astype("boolean")

def sanitize_cols(input_df: pd.DataFrame, input_schema: dict) -> pd.DataFrame:
    """
    Checks data types of columns in a DataFrame and casts if necessary.
    
    Also replaces any blank/missing values with NaN
    according to sentinel flags so that the row may
    be dropped later.

    Parameters:
        input_df (pd.DataFrame): Input DataFrame.
        input_schema (dict): Maps a target data type to a dictionary

    Returns:
        pd.DataFrame: original DataFrame but with properly typed columns.
    """

    # Loop through each data type according to the input_schema
    for key, value in input_schema.items():
        check_fxn = value["check_fxn"]
        cast_type = value["cast_type"]
        cast_fxn = value.get("cast_fxn")
        target_cols = value["target_cols"]
        sentinels = value["sentinels"]

        # Loop through each column that should be that data type
        for col in target_cols:

            # Since schema is universal, not all target cols apply
            # to the input_df
            if col not in input_df.columns:
                continue

            # Exception to schema: loudness may be -1
            if col == "loudness":
                active_sentinels = [s for s in sentinels if s != -1]
            else:
                active_sentinels = sentinels

            # Normalize text before matching sentinels, otherwise a
            # padded blank ("  None  ") survives the replace and is
            # then stripped down to an empty string, which dropna()
            # cannot catch.
            if cast_type == "string":
                input_df[col] = (
                    input_df[col]
                    .astype("string")
                    .str.replace('"', "", regex=False)
                    .str.replace('[“”]', "", regex=True)
                    .str.replace(r"\s+", " ", regex=True)
                    .str.strip()
                )

            # Replace sentinels (blank/missing values) with NaN
            if active_sentinels:
                input_df[col] = input_df[col].replace(active_sentinels, np.nan)

            # Cast column if necessary
            if not check_fxn(input_df[col]):
                if cast_fxn is not None:
                    input_df[col] = cast_fxn(input_df[col])
                else:
                    input_df[col] = input_df[col].astype(cast_type)
                assert check_fxn(input_df[col]), f"Failed to cast column '{col}' to type '{cast_type}'."

    # Return
    return input_df

#####################################
# LOAD RAW DATA
#####################################

# Validate raw data directory path 
RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "raw_data"
dir_exists(input_dir=RAW_DATA_DIR)

# Validate presence of 2 CSV's in raw data directory
csv_files = sorted(RAW_DATA_DIR.glob("*.csv", case_sensitive=False))
if len(csv_files) != 2:
    raise ValueError(f"Expected exactly 2 CSV files in '{RAW_DATA_DIR}', but found {len(csv_files)}.")

# Load raw data
raw_dfs = []
for file_path in csv_files:
    raw_df = pd.read_csv(file_path)
    raw_dfs.append(raw_df)

#####################################
# ASSIGN AND VALIDATE DF's
#####################################

# df1 should have: album_name, explicit, and track_genre
# df2 should have: year

# Assign df1 and df2
raw_df_cols = set(raw_dfs[0].columns)
if "album_name" in raw_df_cols and \
        "explicit" in raw_df_cols and \
        "track_genre" in raw_df_cols:
    df1 = raw_dfs[0]
    df2 = raw_dfs[1]
else:
    df1 = raw_dfs[1]
    df2 = raw_dfs[0]

assert ("album_name" in df1.columns and
        "explicit" in df1.columns and
        "track_genre" in df1.columns and
        "year" not in df1.columns
        ), "The first data set should have the columns 'album_name', 'explicit', and 'track_genre', but not 'year'."
assert ("year" in df2.columns and
        "album_name" not in df2.columns and
        "explicit" not in df2.columns and
        "track_genre" not in df2.columns
        ), "The second data set should have the column 'year' but not 'album_name' or 'explicit'."

# Drop extra index
df1 = df1.drop(["Unnamed: 0"], axis=1)
df2 = df2.drop(["Unnamed: 0"], axis=1)

# Rename columns in df1 and df2 for uniformity
col_names_map = {
        "artist_name": "artists",
        "track_genre": "genre",
        "album_name": "album",
        "track_name": "song",
        }

df1 = df1.rename(columns=col_names_map)
df2 = df2.rename(columns=col_names_map)

# Check that required columns exist in df1 and df2
missing_req_cols = REQ_COMMON_COLS- set(df1.columns)
assert len(missing_req_cols) == 0, f"The first data set is missing required columns, which are: {missing_req_cols}"
missing_req_cols = REQ_COMMON_COLS - set(df2.columns)
assert len(missing_req_cols) == 0, f"The second data set is missing required columns, which are: {missing_req_cols}"

# Check the data type of columns in df1 and df2
df1 = sanitize_cols(input_df=df1, input_schema=SCHEMA)
df2 = sanitize_cols(input_df=df2, input_schema=SCHEMA)

#####################################
# MERGE DATA AND CLEAN MORE
#####################################

# Merge 'year' from df2 into df1, only keeping common songs
merged_df= pd.merge(df1,
                    df2[["track_id", "year"]],
                    on="track_id",
                    how="inner",
                    )

# NOTE: whitespace and quotes are normalized in sanitize_cols(),
# before sentinels are matched, so no text cleanup is needed here.

# Drop rows with NaN
merged_df = merged_df.dropna()
assert merged_df.isna().sum().sum() == 0, "Failed to resolve NaN values!"

# Separate 'genre' into a separate DataFrame
# Genre was stored per-row, so the same (track_id, genre)
# pair can appear more than once.
genres_df = merged_df[["track_id", "genre"]].copy()
genres_df = genres_df.drop_duplicates()
del merged_df["genre"]

# Only keep rows with unique 'track_id'. Note...
# - Since popularity differs in duplicate rows,
#   keep the row with the highest popularity
# - There were duplicate rows to begin with b/c
#   the genre was stored on a per-row basis.

# Only keep rows with unique 'track_id' where popularity is highest
merged_df = merged_df.sort_values(by="popularity", ascending=False)
merged_df = merged_df.drop_duplicates(subset=["track_id"], keep="first")

# Reset indices
merged_df = merged_df.reset_index(drop=True)
genres_df = genres_df.reset_index(drop=True)

# Numeric values seem to be within a proper range
# print("OUTLIER CHECK")
# print(merged_df.describe().T[["min", "max"]])
# print()

#####################################
# SAVE TO PARQUET FILES
#####################################

# Create and validate save directory
SAVE_DIR= Path(__file__).resolve().parent.parent / "data"
dir_exists(input_dir=SAVE_DIR)
SAVE_SONGS = SAVE_DIR / "spotify_songs.parquet"
SAVE_GENRES = SAVE_DIR / "spotify_genres.parquet"

merged_df.to_parquet(SAVE_SONGS, compression="gzip")
genres_df.to_parquet(SAVE_GENRES, compression="gzip")

