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
import os
import pandas as pd
import numpy as np

#####################################
# CONSTANTS
#####################################

# These schemas help with checking and casting columns
# to correct data types. Refer to check_and_cast_cols().

COMMON_SCHEMA = {
        "string": [
            pd.api.types.is_string_dtype,
            "string",
            ["track_id", "artists", "song"],
            ],
        "category": [
            lambda s: isinstance(s.dtype, pd.CategoricalDtype),
            "category",
            ["time_signature", "mode", "key"],
            ],
        "float": [
            pd.api.types.is_float_dtype,
            "float",
            ["loudness", "speechiness", "acousticness",
             "instrumentalness", "liveness", "valence",
             "tempo", "duration_ms", "danceability",
             "energy",
             ],
            ],
        "int": [
            lambda s: s.dtype == np.int64,
            "int",
            ["popularity"],
            ],
        }

DF1_SCHEMA = {
        "string": [
            pd.api.types.is_string_dtype,
            "string",
            ["album"],
            ],
        "category": [
            lambda s: isinstance(s.dtype, pd.CategoricalDtype),
            "category",
            ["genre"],
            ],
        "bool": [
             pd.api.types.is_bool_dtype,
             "bool",
            ["explicit"],
            ],
        }

DF2_SCHEMA = {
        "int": [
            lambda s: s.dtype == np.int64,
            "int",
            ["year"],
            ],
        }

#####################################
# FUNCITONS 
#####################################

def dir_exists(input_dir: Path):
    """
    Checks that the input directory exists.

    Parameters:
        input_dir (Path): Input directory.

    Returns:
        FileNotFoundError if directory does not exist.
    """
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {input_dir}")

def check_and_cast_cols(input_df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    """
    Checks data types of columns in a DataFrame and casts if necessary.

    Parameters:
        input_df (pd.DataFrame): Input DataFrame.
        schema (dict): Maps a data type to...
                        # [0] function for type checking
                        # [1] cast type
                        # [2] list of column names to
    
    Returns:
        pd.DataFrame: original DataFrame but with properly typed columns.
    """

    # Loop through each data type according to the schema
    for key, (check_fxn, cast_type, cols_to_check) in schema.items():
        # Loop through each column that should be that data type
        for col in cols_to_check:
            # Cast column if necessary
            if not check_fxn(input_df[col]):
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
csv_files = list(RAW_DATA_DIR.glob("*.csv", case_sensitive=False))
if len(csv_files) != 2:
    raise ValueError(f"Expected exactly 2 CSV files in '{RAW_DATA_DIR}', but found {len(csv_files)}.")

# Load raw data
raw_dfs = []
for file_path in os.scandir(RAW_DATA_DIR):
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
req_cols = {
        "track_id", "artists", "song",
        "popularity", "duration_ms", "danceability",
        "energy", "key", "loudness",
        "mode", "speechiness", "acousticness",
        "instrumentalness", "liveness", "valence",
        "tempo", "time_signature",
        }

missing_req_cols = req_cols - set(df1.columns)
assert len(missing_req_cols) == 0, f"The first data set is missing required columns, which are: {missing_req_cols}"
missing_req_cols = req_cols - set(df2.columns)
assert len(missing_req_cols) == 0, f"The second data set is missing required columns, which are: {missing_req_cols}"

# Check the data type of columns in df1 and df2
df1 = check_and_cast_cols(input_df=df1, schema=COMMON_SCHEMA)
df1 = check_and_cast_cols(input_df=df1, schema=DF1_SCHEMA)
df2 = check_and_cast_cols(input_df=df2, schema=COMMON_SCHEMA)
df2 = check_and_cast_cols(input_df=df2, schema=DF2_SCHEMA)

#####################################
# CLEAN THE DATA
#####################################

print("LENGTHS OF DFS")
print(len(merged_df))
print()

# Merge 'year' from df2 into df1, only keeping common songs
merged_df= pd.merge(df1,
                    df2[["track_id", "year"]],
                    on="track_id",
                    how="inner",
                    )

# Clean up extra whitespace and double quotes
for col in merged_df[["song", "album", "artists"]]:
    merged_df[col] = merged_df[col].str.replace('"', "", regex=False)
    merged_df[col] = merged_df[col].str.replace('[“”]', "", regex=True)
    merged_df[col] = merged_df[col].str.replace(r'\s+', ' ', regex=True).str.strip()

# Drop rows with blank/missing values
# 1. Replace value with NaN
# 2. Drop NaN

# Set lists of data types and columns
#BLANKS = {
#        "string": [
#            ["Unknown", "UNKOWN", "null", "None", "none", ""],
#            ["track_id", "artists", "album", "song"]
#            ],
#        "float": [
#            [
#                "energy", "speechiness", "acousticness",
#                "instrumentalness", "liveness", "valence",
#                "tempo", "duration_ms", "danceability",
#                ], # loudness can be negative
#            [-1]
#            ],
#bool_cols = ["explicit"]
#category_cols = ["time_signature", "mode", "key"] # genre is separate
#float_cols = int_cols = ["popularity", "year"]
#
#replace_bladdnk_values(input_df=merged_df,
#                     cols_to_check=
#
#df[string_cols] = df[string_cols].replace(string_sentinels, np.nan)
#df[numeric_cols] = df[numeric_cols].replace(numeric_sentinels, np.nan)

print("VALUE COUNTS")
for col in merged_df.columns:
    print(merged_df[col].value_counts())
print()

# Separate 'genre' into a separate DataFrame
genres_df = merged_df[["track_id", "genre"]].copy()
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

print("NEW LENGTHS OF DFS")
print(len(merged_df))
print(len(set(genres_df["track_id"])))
print(len(genres_df))
print()

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
SAVE_SONGS = SAVE_DIR / "spotify_songs.csv"
SAVE_GENRES = SAVE_DIR / "spotify_genres.csv"

# merged_df.to_parquet(SAVE_SONGS, compression="parquet")
# genres_df.to_parquet(SAVE_GENRES, compression="parquet")

