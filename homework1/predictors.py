"""
Olivia Buchanan
Homework 1: Explore 4 Predictive Models
"""

import spotifysongs
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

########################################################
# SEED AND CONSTANTS
########################################################

# Set seed
np.random.seed(12345)

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
# FUNCTIONS
########################################################

def check_formula(formula_preds: np.ndarray,
                  lr_preds: np.ndarray,
                  ):
    """
    Check whether formula predictions and lr.predict() match.

    Parameters:
        formula_preds (np.ndarray): predictions from the hand-made formula.
        lr_preds (np.ndarray): predictions from the model's predict() call.
    
    Returns:
        None. Print statements.
    """

    print()
    print("Are formula_preds and lr_preds about the same?",
        np.allclose(formula_preds, lr_preds),
        )
    print("Formula Predictions: ")
    print(formula_preds)
    print("Model Predictions: ")
    print(lr_preds)

def evaluate_model(lr: LinearRegression,
                   X_train: np.ndarray,
                   X_test: np.ndarray,
                   y_train: np.ndarray,
                   y_test: np.ndarray,
                   ):
    """
    Evaluates a model by calculating the R^2 and RMSE.

    Parameters:
        lr (Linear Regression): the model to evaluate.
        X_train (np.ndarray): the design matrix of training data.
        X_test (np.ndarray): the design matrix of testing data.
        y_train (np.ndarray): vector of true targets for training.
        y_test (np.ndarray): vector of true targets for testing.

    Returns:
        dict: train and test R^2 and RMSE of the model.
    """

    y_pred_train = lr.predict(X_train)
    y_pred_test = lr.predict(X_test)
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

    return {"r2_train" : r2_train,
            "r2_test" : r2_test,
            "rmse_train" : rmse_train,
            "rmse_test" : rmse_test,
            }

def print_model_evaluation(eval_results: dict):
    """
    Prints dictionary of model evaluation results.

    Parameters:
        eval_results (dict): dictionary of model evaluation results.
    
    Returns:
        None
    """
    print()
    print("Training R^2: {rsq:.3f}".format(rsq=eval_results["r2_train"]))
    print("Test R^2: {rsq:.3f}".format(rsq=eval_results["r2_test"]))
    print()
    print("Training RMSE: {rmse:.3f}".format(rmse=eval_results["rmse_train"]))
    print("Test RMSE: {rmse:.3f}".format(rmse=eval_results["rmse_test"])) 

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
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y,
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
m1_scaler = StandardScaler()
m1_scaler.fit(x1_train)
x1_train_scaled = m1_scaler.transform(x1_train)
x1_test_scaled = m1_scaler.transform(x1_test)

# Combine scaled x1 with intercept from X_train_raw
X_train = np.concatenate([X_train_raw.iloc[:, 0:1], x1_train_scaled],
                         axis=1)
X_test = np.concatenate([X_test_raw.iloc[:, 0:1], x1_test_scaled],
                        axis=1)

# Do linear regression
m1_lr = LinearRegression(fit_intercept=False)
m1_lr.fit(X_train, y_train)

# Convert learned coefs from z-score to raw
# Note: indices in scaler are off-by-one compared to lr
# because lr has intercept while scaler doesn't...
# scaler.mean_[0] --> mean
# scaler.scale_[0] --> std
# lr.coef_[0] --> b0
# lr.coef_[1] --> b1
inter = m1_lr.coef_[0] - (m1_lr.coef_[1] * m1_scaler.mean_[0] / m1_scaler.scale_[0])
slope = m1_lr.coef_[1] / m1_scaler.scale_[0]

# Note: average value for target and b0 should be about the same
# print()
# print("Check: scaled intercept should be about the same as mean in y_train")
# print(np.allclose(lr.coef_[0], y_train.mean()))

# Get formula for "y = (coef * intercept) + (coef * x1)"
print()
print("Formula (no scaled terms):")
print("y = {inter:.3f} + {slope:.3f}*{x1}".format(inter=inter,
                                                  slope=slope,
                                                  x1=X_train_raw.columns[1],
                                                  ))

# My formula (takes raw values) and predict() (takes scaled values) should yield the same results
# formula_preds = inter + slope * x1_test[0:10]
# lr_preds = (m1_lr.predict(X_test[0:10, :])).reshape(-1, 1)
# check_formula(formula_preds=formula_preds, lr_preds=lr_preds)

# Calculate R^2 and RMSE
eval_results = evaluate_model(lr=m1_lr,
                              X_train=X_train,
                              X_test=X_test,
                              y_train=y_train,
                              y_test=y_test,
                              )
print_model_evaluation(eval_results=eval_results)

########################################################
# MODEL 1Q: QUADRATIC REGRESSION
########################################################

print()
print("========== MODEL 1Q: QUADRATIC REGRESSION ==========")

# Extract x1 from raw train and test data
x1_train = X_train_raw.iloc[:, 1:2].to_numpy()
x1_test = X_test_raw.iloc[:, 1:2].to_numpy()

# Make (x1)^2 term from x1
x1_train_sq = np.pow(x1_train, 2)
x1_test_sq = np.pow(x1_test, 2)

# Combine x1 and (x1)^2 into one design matrix X_train
X_train_poly = np.concatenate([x1_train, x1_train_sq],
                                 axis=1)
X_test_poly = np.concatenate([x1_test, x1_test_sq],
                                axis=1)

# Standardize x1 and (x1)^2 into z-scores
m1q_scaler = StandardScaler()
m1q_scaler.fit(X_train_poly)
X_train_scaled = m1q_scaler.transform(X_train_poly)
X_test_scaled = m1q_scaler.transform(X_test_poly)

# Combine scaled x1 and (x1)^2 with intercept from X_train_raw
X_train = np.concatenate([X_train_raw.iloc[:, 0:1], X_train_scaled],
                         axis=1)
X_test = np.concatenate([X_test_raw.iloc[:, 0:1], X_test_scaled],
                        axis=1)

# Do linear regression
m1q_lr = LinearRegression(fit_intercept=False)
m1q_lr.fit(X_train, y_train)

# Convert learned coefs from z-score to raw
# Note: indices in scaler are off-by-one compared to lr
# because lr has intercept while scaler doesn't...
# scaler.mean_[0] --> mean for x1
# scaler.mean_[1] --> mean for (x1)^2
# scaler.scale_[0] --> std for x1
# scaler.scale_[1] --> std for (x1)^2
# lr.coef_[0] --> b0
# lr.coef_[1] --> b1
# lr.coef_[2] --> b2
b0 = m1q_lr.coef_[0] - (m1q_lr.coef_[1] * m1q_scaler.mean_[0] / m1q_scaler.scale_[0]) -\
                      (m1q_lr.coef_[2] * m1q_scaler.mean_[1] / m1q_scaler.scale_[1])
b1 = m1q_lr.coef_[1] / m1q_scaler.scale_[0]
b2 = m1q_lr.coef_[2] / m1q_scaler.scale_[1]

# Get formula for "y = (coef * intercept) + (coef * x1) + [coef * (x1)^2]"
print()
print("Formula (no scaled terms):")
print("y = {b0:.3f} + {b1:.3f}*{x1} + {b2:.3f}*({x1_sq})^2".format(b0=b0,
                                                               b1=b1,
                                                               x1=X_train_raw.columns[1],
                                                               b2=b2,
                                                               x1_sq=X_train_raw.columns[1],
                                                               ))

# My formula (takes raw values) and predict() (takes scaled values) should yield the same results
# formula_preds = b0 + (b1 * x1_test[0:10]) + (b2 * x1_test_sq[0:10])
# lr_preds = (lr.predict(X_test[0:10, :])).reshape(-1, 1)
# check_formula(formula_preds=formula_preds, lr_preds=lr_preds)

# Calculate R^2 and RMSE
eval_results = evaluate_model(lr=m1q_lr,
                              X_train=X_train,
                              X_test=X_test,
                              y_train=y_train,
                              y_test=y_test,
                              )
print_model_evaluation(eval_results=eval_results)

########################################################
# PLOT: MODEL 1 VS MODEL 1Q
########################################################

# Make new figure
fig, ax = plt.subplots(figsize=(13,8))

# Plot the raw training data "x1 vs y"
x1_train = X_train_raw.iloc[:, 1:2].to_numpy()
ax.scatter(x1_train, y_train, alpha=0.3, s=20)

# Get raw x and x^2 to plot trend lines
NUM_XS = 1000
xs = np.linspace(x1_train.min(), x1_train.max(), NUM_XS).reshape(-1, 1)
xs_poly = np.concatenate([xs, xs**2], axis=1)

# Standardize the raw x's for each model
m1_xs_scaled = m1_scaler.transform(xs)
m1q_xs_scaled = m1q_scaler.transform(xs_poly)

# Add intercept to design matrix for each model
m1_xs_matrix = np.concatenate([np.ones((NUM_XS, 1)), m1_xs_scaled],
                              axis=1)
m1q_xs_matrix = np.concatenate([np.ones((NUM_XS, 1)), m1q_xs_scaled],
                               axis=1)

# Plot the Model 1 trend line
preds = m1_lr.predict(m1_xs_matrix)
ax.plot(xs, preds, color="green", label="Model 1")

# Plot the Model 1Q trend line
preds = m1q_lr.predict(m1q_xs_matrix)
ax.plot(xs, preds, color="red", label="Model 1Q")

# Label the plot
ax.legend()
ax.set_xlabel("Duration (sec)")
ax.set_ylabel("Popularity")
ax.set_title("Duration vs Popularity in Spotify Songs", fontsize="large")

# Save the plot
fig.savefig("duration.png", format="png")
plt.close(fig)

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





