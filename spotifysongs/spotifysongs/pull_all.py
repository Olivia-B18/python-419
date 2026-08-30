'''
Olivia Buchanan
Homework 0: Setting up Data Set

ABOUT: This script imports all csv files from the raw_data/ directory into DataFrames, cleans the data, and then saves the processed data as Parquet files to be loaded by load.py 

DATA: raw_data/*
'''

from pathlib import Path
import os
import pandas as pd

def is_csv(file_path: Path = None):
    """
    Check if given file is a csv.
    Source: https://www.geeksforgeeks.org/python/python-loop-through-folders-and-files-in-directory/
    """
    try:
        pd.read_csv(file_path, nrows=1) 
        return True
    except pd.errors.ParserError:
        print("Pandas failed to parse the file as CSV.")
        return False
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return False
    except UnicodeDecodeError:
        print(f"Error: File '{file_path}' is not text-based.")
        return False

###########################
# LOAD RAW DATA
############################

# Configure directory paths and validate them
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
RAW_DATA_DIR = PROJECT_ROOT / "raw_data"
os.path.exists(RAW_DATA_DIR)

# Read all CSV's from the directory raw_data/
raw_dfs = []
for entry in os.scandir(RAW_DATA_DIR):
    if is_csv(entry):
        raw_dfs.append(pd.read_csv(entry))

# Validate raw data: exactly 2 files in proper format
try:
    if len(raw_dfs) != 2:
        raise ValueError("There must be exactly two CSV's of Spotify song data.")
        exit()
    for raw_df in raw_dfs:
        if "track_id" not in raw_df.columns.values:
            raise RuntimeError("The two CSV's of Spotify song data must have a 'track_id' column.")
        if isinstance(raw_df["track_id"].dtype, str):
            raise TypeError("The 'track_id' column must be type str.")
except ValueError as e:
    print(f"Error: {e}")
except RuntimeError as e:
    print(f"Error: {e}")
except TypeError as e:
    print(f"Error: {e}")

# Put raw DataFrames in variables for readability
if len(raw_dfs[0]) < len(raw_dfs[1]):
    df1 = raw_dfs[0]
    df2 = raw_dfs[1]
else:
    df1 = raw_dfs[1]
    df2 = raw_dfs[0]

###########################
# CLEAN THE DATA
###########################

# Merge the data
merged_df= pd.merge(df1,
                    df2[["track_id", "year"]],
                    on="track_id",
                    how="inner",
                    )

# Rename columns for readability
merged_df = merged_df.rename(columns={
    "Unnamed: 0": "index",
    "track_name": "title",
    "track_genre": "genre",
    })

# Drop extra index
merged_df = merged_df.drop(["index"], axis=1)

# Drop any rows with blank values
merged_df = merged_df.dropna()

print(len(merged_df))

# Other cleaning?
# None for titles. Apostrophes don't affect string processing.
# None otherwise. Data was already pretty clean.

# Check for rows with more than one apostrophe and drop them
# dropit_idx = []
# for idx, row in merged_df.iterrows():
#     if row["song_title"].count("'") >= 2:
#         dropit_idx.append(idx)
# merged_df = merged_df.drop(index=dropit_idx)

###########################
# SAVE TO PARQUET FILES
###########################

# Create and validate save directory
SAVE_DIR = PROJECT_ROOT / "data"
os.path.exists(SAVE_DIR)
SAVE_PATH = SAVE_DIR / "spotify_songs.parquet"

# merged_df.to_parquet(SAVE_PATH, compression="parquet")

