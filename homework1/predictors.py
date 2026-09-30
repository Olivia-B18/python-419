"""
Olivia Buchanan
Homework 1: Explore 4 Predictive Models
"""

import spotifysongs
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

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
KEY_COLS = [col for col in SONGS.columns if col.startswith("key_")]
x1_name = "duration_s"
x2_name = "valence"
FEATURES = [x1_name, x2_name]
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

# What are the columns in X?
# intercept, x1, x2, x3, etc. according to FEATURES
# print(X.columns)

# Train/test split
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=0.2,
                                                            shuffle=True)

########################################################
# MODEL 1: LINEAR REGRESSION 
########################################################

print()
print("========== MODEL 1: LINEAR REGRESSION ==========")

# Extract x1 from raw train and test data
x1_train = X_train_raw.iloc[:, 1:2].to_numpy()
x1_test = X_test_raw.iloc[:, 1:2].to_numpy()

# Standardize x1 into z-scores
scaler = StandardScaler()
scaler.fit(x1_train)
x1_train_scaled = scaler.transform(x1_train)
x1_test_scaled = scaler.transform(x1_test)

# Combine scaled x1 with intercept from X_train_raw
X_train = np.concatenate([X_train_raw.iloc[:, 0:1],
                          x1_train_scaled],
                          axis=1,
                          )
X_test = np.concatenate([X_test_raw.iloc[:, 0:1],
                         x1_test_scaled],
                        axis=1,
                        )

# Do linear regression
lr = LinearRegression(fit_intercept=False)
lr.fit(X_train, y_train)

# Convert learned coefs from z-score to raw
# scaler.mean_[0] --> mean
# scaler.scale_[0] --> std
# lr.coef_[0] --> b0
# lr.coef_[1] --> b1
inter = lr.coef_[0] - (lr.coef_[1] * scaler.mean_[0] / scaler.scale_[0])
slope = lr.coef_[1] / scaler.scale_[0]

# Average value for target and b0 should be about the same
# print("Check: scaled intercept should be about the same as mean in y_train")
# print(np.allclose(lr.coef_[0], y_train.mean()))

# Get formula for "y = (coef * intercept) + (coef * x1)"
print()
print("Formula (no scaled terms):")
print("y = {inter:.3f} + {slope:.3f}*{x1}".format(inter=inter,
                                                 slope=slope,
                                                 x1=X_train_raw.columns[1],
                                                 ))

# My formula and predict() should yield the same results
# formula_preds = inter + slope * x1_test[0:10]
# lr_preds = (lr.predict(X_test[0:10, :])).reshape(-1, 1)
# 
# print()
# print("Are formula_preds and lr_preds about the same?",
#       np.allclose(formula_preds, lr_preds),
#       )
# print("Formula Predictions: ")
# print(formula_preds)
# print("Model Predictions: ")
# print(lr_preds)

# Calculate R^2 and RMSE
y_pred_train = lr.predict(X_train)
y_pred_test = lr.predict(X_test)
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

print()
print("Training R^2: {rsq:.3f}".format(rsq=r2_train))
print("Test R^2: {rsq:.3f}".format(rsq=r2_test))
print()
print("Training RMSE: {rmse:.3f}".format(rmse=rmse_train))
print("Test RMSE: {rmse:.3f}".format(rmse=rmse_test))

########################################################
# MODEL 1Q: QUADRATIC REGRESSION
########################################################



########################################################
# MODEL 2: TWO NUMERICAL FEATURES
########################################################



########################################################
# MODEL 3: ADD CATEGORICAL FEATURE
########################################################

# Note: must leave out key_0 (key of C) to
# create a baseline. Otherwise, all row-wise sums
# yield 1, and coefficients become useless. Now,
# a baseline will create an idea that key_1 is more
# or less popular than C songs.





