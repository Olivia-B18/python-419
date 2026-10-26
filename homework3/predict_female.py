"""
Olivia Buchanan
Homework 3: Feature Engineering
"""

import string
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, KFold
from sklearn.metrics import log_loss

# TODO: add hypothesis testing

###################################################################################
# CONSTANTS
###################################################################################

# Seed
SEED = 17

# File names
TRAIN_FILENAME = "train.csv"
TEST_FILENAME = "test.csv"
SUBMISSION_FILENAME = "submission.csv"

# Cross validation settings
SCORING = ["neg_log_loss"]
NUM_CV = 10

# Define alphabet
ALPHABET = list(string.ascii_lowercase)

# Features
TARGET = "female"
NAME = "name"
ID = "id"
NUM_FEATURES = [
    "length",
    "num_vowels",
    "vowel_ratio",
]
CAT_FEATURES = [
    "first_letter",
    "last_letter",
    "ends_n",
    "ends_vowel",
]
LETTER_FEATURES = [
    "first_letter",
    "last_letter",
]
FINAL_FEATURES = {
    "num": NUM_FEATURES,
    "cat": CAT_FEATURES,
}

# Models to evaluate
SCHEMAS = [
    {
        "name": "ALL_NUMERIC",
        "num": NUM_FEATURES,
        "cat": [],
    },
    {
        "name": "ALL_CATEGORICAL",
        "num": [],
        "cat": CAT_FEATURES,
    },
    {
        "name": "ALL FEATURES",
        "num": NUM_FEATURES,
        "cat": CAT_FEATURES,
    },
    {
        "name": "FINAL_FEATURES",
        "num": FINAL_FEATURES["num"],
        "cat": FINAL_FEATURES["cat"],
    },
]

###################################################################################
# FUNCTIONS
###################################################################################

def create_features(initial_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a DataFrame of features for predicting gender of names.

    Parameters:
        initial_df (pd.DataFrame): the initial DataFrame with just a "name" column.
    
    Returns:
        pd.DataFrame. Full training data features.
    """

    # Make copy of initial df to modify and then return
    df = initial_df.copy()

    # "Clean" the initial df, make names lower case
    df[NAME] = df[NAME].str.lower()

    # Add numeric features
    df["length"] = df[NAME].str.len()
    df["num_vowels"] = df[NAME].str.count("[aeiouy]")
    df["vowel_ratio"] = df["num_vowels"] / df["length"]

    # Add categorical features
    df["first_letter"] = df[NAME].str[0]
    df["last_letter"] = df[NAME].str[-1]
    df["ends_n"] = df[NAME].str[-1].isin(list("n")).astype("int")
    df["ends_vowel"] = df[NAME].str[-1].isin(list("aeiouy")).astype("int")

    # Remove "name" column from the df
    df = df.drop(columns=[NAME])

    # Return
    return df

def build_pipeline(num: list, cat: list) -> Pipeline:
    """
    Build a pipeline given the numerical and categorical features.

    Parameters:
        num (list): numeric feature names.
        cat (list): categorical feature names.

    Returns:
        Pipeline.
    """

    # Split categorical features into letter features and binary features
    letter = [feature for feature in cat if feature in LETTER_FEATURES]
    binary = [feature for feature in cat if feature not in LETTER_FEATURES]

    # Create feature transformers
    transformers = []
    if num:
        transformers.append(("num", StandardScaler(), num))
    if letter:
        # Do not want to compare "a"-ness, so leave all 26 columns
        transformers.append(("letter", OneHotEncoder(categories=[ALPHABET] * len(letter),
                                                     handle_unknown="ignore"),
                                                     letter,
                                                     ))
    if binary:
        # Want to compare binary results, so best drop one output
        transformers.append(("binary", OneHotEncoder(categories=[[0, 1]] * len(binary),
                                                     drop="if_binary"),
                                                     binary,
                                                     ))

    # Return pipeline
    return Pipeline([
        ("prep", ColumnTransformer(transformers)),
        ("lr", LogisticRegression(fit_intercept=True,   # add col of 1's
                                  C=1.0,                # strength of regularization (smaller is stronger)
                                  l1_ratio=1,           # lasso regression
                                  solver="liblinear",   # optimization algorithm, fast for small datasets
                                  random_state=SEED,    # set seed
                                  )),
    ])

def evaluate_cross_validation(model_schema: dict,
                              target: pd.Series,
                              train: pd.DataFrame,
                              ) -> dict:
    """
    Evaluate logistic regression of a model.

    Parameters:
        model_schema (dict): schema of model names and features.
        target (pd.Series): the target to predict.
        train (pd.DataFrame): the features for training.

    Returns:
        Dict. Returns model results as a dictionary.
    """

    # Create Pipeline object
    model_pipeline = build_pipeline(num=model_schema["num"],
                                    cat=model_schema["cat"],
                                    )

    # Get list of features
    features = model_schema["num"] + model_schema["cat"]

    # Do cross validation for a model
    cv_output = cross_validate(
            model_pipeline,
            train[features],
            target,
            cv=KFold(NUM_CV, shuffle=True, random_state=SEED),
            scoring=SCORING,
            return_estimator=True,
            )

    # Return scores
    results = {}
    for metric_name, values in cv_output.items():
        if "test_neg_log_loss" in metric_name:
            # CV uses neg_log_loss but report as positive
            results["log_loss"] = -values
        elif "test_" in metric_name:
            results[metric_name] = values

    return results

def print_cv_results(models: list, results: list) -> None:
    """
    Print the results from cross evaluation of a model.

    Parameters:
        models (list): schemas of model names and features.
        results (list): parallel list of model results.
    
    Returns:
        None. Print statements.
    """

    # Find range and average of log loss spread for each model
    model_stats = []
    for model in results:
        stats = {}
        for metric_name, values in model.items():
            stats[metric_name] = np.average(values)
            stats["range "+metric_name] = max(values) - min(values)
        model_stats.append(stats)

    # Get model log loss stats as a df
    stats_df = pd.DataFrame(model_stats, index=[model["name"] for model in models])

    # Print results
    print()
    print("=" * 10, "LOG LOSS RESULTS", "=" * 10)
    for metric_name in stats_df.columns:
        print()
        print("Smallest "+metric_name+":", round(stats_df[metric_name].min(), 6))
        print("Model:", stats_df[metric_name].idxmin())
    print()
    print(stats_df)

def get_betas(model_pipeline: Pipeline) -> tuple:
    """
    Get intercept and coefficients of a model.

    Parameters:
        model_pipeline: Pipeline for a model.

    Returns:
        Tuple: (intercept, coefs).
    """
    
    # Get feature names for indexing
    feature_names = model_pipeline.named_steps["prep"].get_feature_names_out()

    # Get intercept and coefficients
    intercept = model_pipeline.named_steps["lr"].intercept_[0]
    coefs = pd.Series(model_pipeline.named_steps["lr"].coef_[0], index=feature_names)

    # Return
    return (intercept, coefs)

def validate_submission(submission: pd.DataFrame,
                        len_test: int,
                        ) -> True:
    """
    Assert statements verify that submission is in valid format.

    Parameters:
        submission (pd.DataFrame): the DataFrame of mushroom
            id's and probabilities to submit.
        len_test (int): the length of the test data.

    Returns:
        True if the submission is valid. Else, assert error.
    """
    
    # Assert the df is valid
    assert len(submission) == len_test, "Submission does not have predictions \
        for all rows in test data."
    assert list(submission.columns) == ["id", "female"], "Submission columns \
        must be only 'id' and 'female'."
    assert submission["id"].is_unique, "Submission id's must be unique."
    assert submission["female"].max() <= 1, "Submission probabilities must \
        not exceed 1."
    assert submission["female"].min() >= 0, "Submission probabilities must \
        not fall below 0."

    # Write submission to csv
    submission.to_csv(SUBMISSION_FILENAME, index=False)

    # Read csv to check that index is missing
    written = pd.read_csv(SUBMISSION_FILENAME)
    assert list(written.columns) == ["id", "female"], "Submission file must \
        not include the pandas index."

    return True

###################################################################################
# LOAD DATA AND CREATE FEATURES
###################################################################################

# Load datacoefficients
train = pd.read_csv(TRAIN_FILENAME)
test = pd.read_csv(TEST_FILENAME)

# Get target and name columns
target = train[TARGET]

# Get features
train = create_features(initial_df=train[[NAME]])
test = create_features(initial_df=test[[ID, NAME]])

# Print training feature
# print()
# print("=" * 10, "RAW TRAINING DATA", "=" * 10)
# print(train.head(5))

###################################################################################
# THE BASELINE: PREDICT THE BASELINE
###################################################################################

# NOTE: always predict 0.5 -> always log loss of 0.693147.

# # Baseline for any logistic regression is 0.5 probability
# BASELINE = 0.5

# # Evaluate baseline model
# evaluate_basic_model(model_name="BASELINE", prob=BASELINE, target=target)

###################################################################################
# THE PRIOR: PREDICT MAJORITY CLASS
###################################################################################

# NOTE: turns out that there are an equal number of male and
# female names in the train set, so prior == baseline.

# # Calculate the prior: num_female / total
# prior = len(target[target == 1]) / len(target)

# # Evaluate the prior
# evaluate_basic_model(model_name="PRIOR", prob=prior, target=target)

###################################################################################
# DO LOGISTIC REGRESSION
###################################################################################

results = []

for model in SCHEMAS:
    scores = evaluate_cross_validation(model_schema=model,
                                       target=target,
                                       train=train,
                                       )
    results.append(scores)

print_cv_results(models=SCHEMAS,
                 results=results,
                 )

###################################################################################
# PREDICT ON TEST SET
###################################################################################

# Create final model pipeline object
model_pipeline = build_pipeline(num=FINAL_FEATURES["num"],
                                cat=FINAL_FEATURES["cat"],
                                )

# Combine FINAL_FEATURES into a list
list_final_features = FINAL_FEATURES["num"] + FINAL_FEATURES["cat"]

# Actually train the model on training data
model_pipeline.fit(train[list_final_features], target)

# Scale and encode test data, predict probabilities on test data
p_female = model_pipeline.predict_proba(test[list_final_features])[:, 1]

# Create df for submission
predictions = pd.DataFrame({
    "id": test["id"],
    "female": p_female,
})

# Get the coefficients of the model
intercept, coefs = get_betas(model_pipeline=model_pipeline)

# Print final features picked
print()
print("=" * 10, "FINAL MODEL ANALYSIS", "=" * 10)
print()
print("Final features:")
print("num:", FINAL_FEATURES["num"])
print("cat:", FINAL_FEATURES["cat"])

# Print coefficients of the model
# Note:
# - The sign indicates pos/neg correlation.
# - The magnitude indicates how much the target is
#       influenced by a one-unit change in a feature,
#       holding other features constant.
print()
print("Intercept:", round(intercept, 6))
print()
print("Coefficients of the final model are:")
print(coefs.sort_values())

###################################################################################
# MAKE SUBMISSION
###################################################################################

# Assert the df is valid and make submission
if validate_submission(submission=predictions, len_test=len(test)):
    print()
    print("=" * 10, "END PROGRAM", "=" * 10)
    print()
    print("Successfully wrote predictions to:", SUBMISSION_FILENAME)
    print()
