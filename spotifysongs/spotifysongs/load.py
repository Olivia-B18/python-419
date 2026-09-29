'''
Olivia Buchanan
Homework 0: Setting up Data Sets

DATA: data/*
'''

from pathlib import Path
import os
import pandas as pd


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
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"No data/ directory. Run pull_all.py first: {DATA_DIR}")

    # Load data
    songs_df = pd.read_parquet(DATA_DIR / "spotify_songs.parquet")

    # Return
    return {
        "songs": songs_df,
    }

