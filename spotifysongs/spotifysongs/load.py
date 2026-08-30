'''
Olivia Buchanan
Homework 0: Setting up Data Sets

DATA: data/*
'''

from pathlib import Path
import os
import pandas as pd

def load():
    """
    Load data from parquet files in data/ directory.
    """
    
    # Configure paths and validate them
    CURRENT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = CURRENT_DIR.parent
    DATA_DIR = PROJECT_ROOT / "data"
    os.path.exists(DATA_DIR)

    # Load data
    songs_df = pd.read_parquet(DATA_DIR / "spotify_songs.parquet")

    # Return
    return {
        "songs": songs_df,
    }

