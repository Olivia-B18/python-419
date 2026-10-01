# DATA 419 Repo

## About

This repo is an exploration of data mining according to UMW's DATA 419
Data Mining course with Professor Stephen Davies in the Fall 2026
semester.

The data set I study consists of Spotify songs sourced from Kaggle.
Before I began analyzing my data, I created a package `spotifysongs`
to clean and prepare it.

## Installation

1. Create and activate a virtual environment in Python 3.12 (I used conda).
    ```
    conda create -n spotifysongs-env python=3.12
    conda activate spotifysongs-env
    ```
2. Install the dependencies.
    ```
    pip install dependencies.txt
    ```
3. Clone this repo into your desired directory.
4. Follow the installation instructions in `spotifysongs/README.md`
to get the data and dependencies.
5. Go to any `homework` directory and run the main script for that experiment.

## Tour of Directories

* `spotifysongs`: directories to hold the raw and processed data
    sets as well as the `spotifysongs` package.
* `homework`: each homework investigates something different about
    my data set.
* `practice`: assignments I was given to brush up on Python.
* `notes`: my personal notes as markdown files.
* `data419`: the class repo
* `write_dependencies.py`: reads any imports in the project and logs the
    package with pins. Direct the terminal output to a file.

## AI Usage

This project was developed in tandem with Claude Code (Opus 5) and Gemini
Flash 3.6. AI was used for API queries, debugging, co-writing code, and more.
