'''
Olivia Buchanan
Homework 0: Setting up Data Sets

DATA: data/*
'''

from pathlib import Path
import os
import pandas as pd

############################################
# CONSTANTS
############################################

# Columns that pull_all.py stores as 'category'.
# Parquet files only save category types correctly for
# string-looking values, not integer-looking ones.
# Must re-cast.
CATEGORY_COLS = [
    "key",
    "mode",
    "time_signature",
    "genre",
]

#############################################
# FUNCTIONS
###########################################

def load() -> dict:
    """
    Load data from parquet files in data/ directory.

    Parameters:
        - None

    Returns:
        dict: Dictionary of DataFrames of spotify songs.
    """
    
    # Configure paths and validate them
    CURRENT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = CURRENT_DIR.parent
    DATA_DIR = PROJECT_ROOT / "data"
    os.path.exists(DATA_DIR)

    # Load data
    songs_df = pd.read_parquet(DATA_DIR / "spotify_songs.parquet")
    genres_df = pd.read_parquet(DATA_DIR / "spotify_genres.parquet")

    # Restore categorical dtypes lost in the parquet round-trip
    for df in (songs_df, genres_df):
        for col in CATEGORY_COLS:
            if col in df.columns:
                df[col] = df[col].astype("category")

    # Return
    return {
        "songs": songs_df,
        "genres": genres_df,
    }

