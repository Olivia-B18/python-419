# The Package `spotifysongs`

## About

This file details where I got my data sets of Spotify songs from,
how to install my `spotifysongs` package, and how to run the package.

## Original Data Source 

I made a free account with Kaggle and downloaded two CSV's of
Spotify songs from the following URL's:

* https://www.kaggle.com/datasets/priyamchoksi/spotify-dataset-114k-songs?resource=download
* https://www.kaggle.com/datasets/amitanshjoshi/spotify-1million-tracks/data

## Installation

These instructions walk you through manually making any necessary
directories and installing dependencies. From the project root,
run the following commands in your terminal:

```
mkdir -p spotifysongs/raw_data

# MANUAL: unzip and move the downloaded CSV's into spotifysongs/raw_data

pip install -e ./spotifysongs

python -m spotifysongs.pull_all
```

## Calling the Load Function

Run Python while in a homework directory, not the project root.
Your Python script will call the load function in `spotifysongs`
like so:

```
import spotifysongs

tables = spotifysongs.load()

# Alternatively: use globals().update(spotifysongs.load())
# and get var "songs" directly
```

For example, go to `homework0` and run the `test_load.py` script.
This executes the package's load function to load parquet files.
You should see the `songs` DataFrame print successfully.

```
cd homework0
python test_load.py
```

## Overview of Generated `songs` DataFrame

`songs` is the main and only data set, containing about 36,000 unique
track id's and info on each song, including: artists, album,
popularity, various audio features, and more. There are four kinds of columns
in the `songs` data set:

1. **Genres.** Genres are many-hot encoded directly into `songs` as
113 columns named `genre_<name>` (i.e. `genre_edm`, `genre_pop`, ...).
A song may belong to more than one genre, so more than one of those
columns may be 1 for a given row.

2. **Time Signature.** Time signature is one-hot encoded as 5 columns
named `time_signature_<name>` (i.e. `time_signature_3`,
`time_signature_4`, ..., `time_signature_7`).

3. **Key.** Keys are recorded using pitch notation and are one-hot
encoded as 12 columns named `key_<name>` f(i.e. `key_0`, `key_1`, ...,
`key_11`).

4. **Song Info.** Song info contains all other info on a track. It
comes from the first data set (`kaggle_114k_songs.csv`). The second
data set contributes only ```year```; where the two disagree on a
column such as popularity or acousticness, the first data set wins.
If the same track appears more than once since it was released in
multiple albums, the version with the highest popularity wins.
