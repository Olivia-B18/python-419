# The Package ```spotifysongs```

## About

This file details where I got my data sets of Spotify songs from, how to install my ```spotifysongs``` package, and how to run the package.

## Original Data Source 

I made a free account with Kaggle and downloaded two CSV's of Spotify songs from the following URL's:

* https://www.kaggle.com/datasets/priyamchoksi/spotify-dataset-114k-songs?resource=download
* https://www.kaggle.com/datasets/amitanshjoshi/spotify-1million-tracks/data

## Installation

These instructions walk you through manually making any necessary directories and installing dependencies. From the project root, run the following commands in your terminal:

```
mkdir -p spotifysongs/raw_data

# MANUAL: unzip and move the downloaded CSV's into spotifysongs/raw_data

pip install -e ./spotifysongs

python -m spotifysongs.pull_all
```

## Calling the Load Function

Run Python while in a homework directory, not the project root. Your Python script will call the load function in ```spotifysongs``` like so:

```
import spotifysongs

tables = spotifysongs.load()
```

For example, go to ```homework0``` and run the ```test_load.py``` script. This executes the package's load function to load parquet files. You should see the ```songs``` and ```genres``` DataFrames print successfully.

```
cd homework0
python test_load.py
```

## Overview of Generated DataFrames

* ```songs```: the is the main data set containing about 36,000 unique track id's and info on each song, including artists, album, popularity, and audio features.
* ```genres```: this is a related data set containing about 45,000 track id / genre pairs. The same song will appear more than once if it falls under more than one genre.