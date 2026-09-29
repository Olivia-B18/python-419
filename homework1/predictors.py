"""
Olivia Buchanan
Homework 3: Explore 4 Predictive Models
"""

import spotifysongs
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 

########################################################
# SEED AND CONSTANTS
########################################################

# Set seed
np.random.seed(12345)

# Get "songs" DataFrame
globals().update(spotifysongs.load())

# Constants
TABLE = spotifysongs.load()
SONGS = TABLE["songs"]
TARGET = "popularity"
KEY_COLS = [col for col in songs.columns if col.startswith("key_")]
FEATURES = ["duration_ms", "valence"]
FEATURES.extend(KEY_COLS)

########################################################
# SETUP: TARGET, FEATURES, TRAIN/TEST SPLIT
########################################################

# Get target vector y and design matrix X
y = SONGS[TARGET]
X = SONGS[FEATURES]

# Add intercept x0 (col of 1s) to design matrix X
ones = np.ones((len(X), 1))
X.insert(0, "intercept", ones)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(X)
print(X.columns)

########################################################
# MODEL 1: LINEAR REGRESSION 
########################################################

# Standardize into z-scores
scaler = StandardScaler()
X_train_numeric = scaler.fit(X_train.iloc[:, 1:])


########################################################
# MODEL 1Q: QUADRATIC REGRESSION
########################################################



########################################################
# MODEL 2: TWO NUMERICAL FEATURES
########################################################



########################################################
# MODEL 3: ADD CATEGORICAL FEATURE
########################################################





