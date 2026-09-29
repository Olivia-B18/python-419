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
from sklearn.preprocessing import OneHotEncoder

#####################################
# SET PANDAS DISPLAY OPTIONS
#####################################

pd.set_option("display.max.columns",80)
pd.set_option("display.width",240)

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
            "album",
            "genre",
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
            "year",
            "time_signature",
            "mode",
            "key",
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

            # Exception to schema: time_signature is only valid for 3-7
            if col == "time_signature":
                invalid_time_sig_vals = list(
                        key
                        for key in input_df["time_signature"].value_counts().keys()
                        if key not in range(3, 8)
                        )
                input_df["time_signature"] = input_df["time_signature"].replace(invalid_time_sig_vals, np.nan)

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

# Clean df1 and df2 before merging:
# - check column data types (cast if necessary)
# - change blank/missing values to NaN to drop later
df1 = sanitize_cols(input_df=df1, input_schema=SCHEMA)
df2 = sanitize_cols(input_df=df2, input_schema=SCHEMA)

#####################################
# SEPARATE GENRES FROM DF1
#####################################

# Note: the "genre" column will later be
# many-encoded. We must detach it from the
# main df to maintain index order later.

genre_pairs = df1[["track_id", "genre"]].copy()
genre_pairs = genre_pairs.dropna().drop_duplicates()
df1 = df1.drop(columns=["genre"])

#####################################
# DROP NA AND DE-DUPLICATE
#####################################

# Remove missing values before merge
df1 = df1.dropna()

# Only keep the track_id with the highest popularity
df1 = df1.sort_values(by="popularity", ascending=False)
df1 = df1.drop_duplicates(subset=["track_id"], keep="first")
assert df1["track_id"].is_unique, "df1 still holds duplicate track_id's!"

#####################################
# MERGE DATA
#####################################

# Extract "year" from df2
df2_year = df2[["track_id", "year"]]
df2_year = df2_year.dropna()
df2_year = df2_year.drop_duplicates(subset=["track_id"], keep="first")

# Merge df1 and df2 on an inner join for only common track_id's
merged_df = pd.merge(df1,
                     df2_year,
                     on="track_id",
                     how="inner",
                     )

# Drop NaN from sanitization
merged_df = merged_df.dropna()

# Convert Int64 (nullable type) back to int64
for col in SCHEMA["int"]["target_cols"]:
    merged_df[col] = merged_df[col].astype(int)

# Asserts
assert merged_df.isna().sum().sum() == 0, "Failed to resolve NaN values!"
assert merged_df["track_id"].is_unique, "merged_df holds duplicate track_id's!"

# Reset the index
merged_df = merged_df.reset_index(drop=True)

#####################################
# MULTI-LABEL ENCODE GENRES
#####################################

# Restrict (track_id, genre) pairs to songs that survived the inner join
genre_pairs = genre_pairs[genre_pairs["track_id"].isin(merged_df["track_id"])]

# ONE-HOT ENCODING VS MULTI-LABEL PROBLEM?
# One-hot encoding means there is a single 1 in a row of
# otherwise 0's, but I can't one-hot encode genres directly
# because I want to see a mix of 1's and 0's in a single
# row for each genre a song falls under. This is a
# multi-label problem.

# Create a pair_matrix where each row represents a single
# (track_id, genre) pair using a sequence of 0's and a single 1
encoder = OneHotEncoder(sparse_output=False, dtype=np.int8)
pair_matrix = encoder.fit_transform(genre_pairs[["genre"]])

# Extract all the genres available and prefix with "genre_"
genre_cols = encoder.get_feature_names_out(["genre"])

# Shove pair_matrix into a DataFrame using track_id as index
genres_df = pd.DataFrame(pair_matrix,
                          columns=genre_cols,
                          index=genre_pairs["track_id"],
                          )

# Group by track_id's and take max for each genre col
genres_df = genres_df.groupby(level=0).max()

# Undo alphabetical order of groupby() using reindex()
genres_df = genres_df.reindex(merged_df["track_id"], fill_value=0)

# Overwrite the index for cleaner concatenation
genres_df.index = merged_df.index

# Add genres_df to merged_df
merged_df = pd.concat([merged_df, genres_df], axis=1)

assert len(merged_df) == len(genres_df), "Genre block lost or gained rows!"

#####################################
# ONE-HOT ENCODE TIME_SIGNATURE
#####################################

# Get "time_signature" column
time_sigs = merged_df[["time_signature"]]

# Valid time signatures are [3, 7] 
encoder = OneHotEncoder(categories=[[3, 4, 5, 6, 7]],
                        sparse_output=False,
                        dtype=np.int8,
                        )
time_sig_matrix = encoder.fit_transform(time_sigs)

# Extract all the time_signatures available and prefix with "time_signature_"
time_sig_cols = encoder.get_feature_names_out(["time_signature"])

# Rows line up one-to-one with merged_df, so reuse its index
time_sigs_df = pd.DataFrame(time_sig_matrix,
                           columns=time_sig_cols,
                           index=merged_df.index,
                           )

# Add the one-hot encoded time signature columns
merged_df = pd.concat([merged_df, time_sigs_df], axis=1)

# Drop the original "time_signature"
merged_df = merged_df.drop(columns=["time_signature"])

assert (time_sigs_df.sum(axis=1) == 1).all(), "Each song should have exactly one time signature!"

#####################################
# ONE-HOT ENCODE KEY
#####################################

# Get "key" column
keys = merged_df[["key"]]

# Valid time signatures are [0, 11] 
encoder = OneHotEncoder(categories=[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]],
                        sparse_output=False,
                        dtype=np.int8,
                        )
keys_matrix = encoder.fit_transform(keys)

# Extract all the keys available and prefix with "key_"
key_cols = encoder.get_feature_names_out(["key"])

# Rows line up one-to-one with merged_df, so reuse its index
keys_df = pd.DataFrame(keys_matrix,
                       columns=key_cols,
                       index=merged_df.index,
                       )

# Add the one-hot encoded keys columns
merged_df = pd.concat([merged_df, keys_df], axis=1)

# Drop the original "time_signature"
merged_df = merged_df.drop(columns=["key"])

assert (keys_df.sum(axis=1) == 1).all(), "Each song should have exactly one key!"

#####################################
# SAVE TO PARQUET FILES
#####################################

# Create and validate save directory
SAVE_DIR= Path(__file__).resolve().parent.parent / "data"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
dir_exists(input_dir=SAVE_DIR)

# Create parquet file path(s)
SAVE_SONGS = SAVE_DIR / "spotify_songs.parquet"

# Save to parquet files path(s)
merged_df.to_parquet(SAVE_SONGS, compression="gzip")

