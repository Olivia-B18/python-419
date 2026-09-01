the original data source;
the source URL, API, or acquisition procedure;
any account or credential requirements, without including secrets;
what each generated table represents;
how to install the package;
how to run pull_all.py;
how to call load();
any known limitations or manual steps.

# The Package ```spotifysongs```

## About

This file details where I got my data sets of Spotify songs from, how to install my ```spotifysongs``` package, and how to run the package.

## Original Data Source 

I made a free account with Kaggle and downloaded two CSV's of Spotify songs from the following URL's:

* https://www.kaggle.com/datasets/priyamchoksi/spotify-dataset-114k-songs?resource=download
* https://www.kaggle.com/datasets/amitanshjoshi/spotify-1million-tracks/data

## Installation

These instructions walk you through manually making any necessary directories and installing dependencies. From the project root, run the following commands:

```
mkdir spotifysongs/raw_data

mkdir spotifysongs/data

pip install ./spotifysongs

python -m spotifysongs.pull_all
```

## Loading the Data

Run the test script in ```homework0```. This executes the package's load function to load parquet files. You should see the ```songs``` and ```genres``` DataFrames print successfully.

```
cd homework0
python test_load.py
```

## Overview of Generated DataFrames

* ```songs```: the is the main data set containing about 36,000 unique track id's and info on each song, including artists, album, popularity, and audio features.
* ```genres```: this is a related data set containing about 45,000 track id / genre pairs. The same song will appear more than once if it falls under more than one genre.