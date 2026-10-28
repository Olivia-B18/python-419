"""
Olivia Buchanan
Homework 3: Feature Engineering
"""

import re
import string
import pandas as pd
import numpy as np
from scipy.stats import pointbiserialr, chi2_contingency
from scipy.stats.contingency import association
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, KFold

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

# Define letter things to use
ALPHABET = list(string.ascii_lowercase)
VOWEL_PAIRS = ['ue', 'ie', 'ae', 'ee', 'oe', 'ui', 'ii', 'ai', 'ei', 'oi', 'ua',
               'ia', 'aa', 'ea', 'oa', 'uo', 'io', 'ao', 'eo', 'oo', 'uu', 'iu',
               'au', 'eu', 'ou']

# Features
TARGET = "female"
NAME = "name"
ID = "id"
NUM_FEATURES = [
    # NOTE: this is redundant since vowel_ratio * len yields this info.
    "num_vowels",

    "length",
    "vowel_ratio",
]
LETTER_FEATURES = [
    "first_letter",
    "last_letter",
]
BINARY_FEATURES = [
    # NOTE: these two are redundant because of "last_letter"
    "ends_n",
    "ends_vowel",

    # NOTE: these two each have low association with target
    # and only 50/50 rate of "female-ness". Not worth it.
    "has_vowel_pair",
    "has_cons_cluster",

    # NOTE: the following are the most female/male suffices,
    # but they provide mostly redundant info from "last_letter"
    "ends_tte",
    "ends_ia",
    "ends_on",
]
CAT_FEATURES = LETTER_FEATURES + BINARY_FEATURES
FINAL_FEATURES = {
    "num": ["length", "vowel_ratio"],
    "cat": ["first_letter", "last_letter"],
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
        "name": "VOWEL_RATIO AND LAST_LETTER",
        "num": ["vowel_ratio"],
        "cat": ["last_letter"],
    },
    {
        "name": "WITH LENGTH",
        "num": ["length", "vowel_ratio"],
        "cat": ["last_letter"],
    },
    {
        "name": "WITH FIRST LETTER",
        "num": ["vowel_ratio"],
        "cat": ["first_letter", "last_letter"],
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

def find_vowel_pair(s: str):
    """
    Check if a string has a vowel pair.
    Source: https://stackoverflow.com/questions/77768248/i-am-trying-to-make-a-diphthong-counter-and-i-cannot-seem-to-get-it-count-all-d

    Parameters:
        s (str): input string.
    
    Returns:
        Boolean.
    """

    for d in VOWEL_PAIRS:
        if d in s:
            return 1
    return 0

def find_cons_cluster(s: str):
    """
    Check for consonant cluster (double consonant).
    Source: https://stackoverflow.com/questions/27744882/finding-consecutive-consonants-in-a-word

    Parameters:
        s (str): input string.
    
    Returns:
        Boolean.
    """
    output = re.findall(r'[bcdfghjklmnpqrstvwxz]+', s, re.IGNORECASE)
    for entry in output:
        if len(entry) > 1:
            return 1
    return 0

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
    df["has_vowel_pair"] = df[NAME].apply(find_vowel_pair)
    df["has_cons_cluster"] = df[NAME].apply(find_cons_cluster)
    df["ends_tte"] = df[NAME].str.endswith("tte").astype("int")
    df["ends_ia"] = df[NAME].str.endswith("ia").astype("int")
    df["ends_on"] = df[NAME].str.endswith("on").astype("int")

    # Remove "name" column from the df
    df = df.drop(columns=[NAME])

    # Return
    return df

def evaluate_correlation(num: list,
                         cat: list,
                         target: pd.Series,
                         train: pd.DataFrame,
                         ):
    """
    Check for correlation between features and the target.
    Calculate r-value and make visuals.

    Parameters:
        num (list): numeric features.
        cat (list): categorical features.
        target (pd.Series): target.
        train (pd.DataFrame): all training features.
    
    Returns:
        None. Print statements.
    """

    ### SETUP ###

    # Separate categorical features into letter and binary
    letter = [feat for feat in cat if feat in LETTER_FEATURES]
    binary = [feat for feat in cat if feat not in LETTER_FEATURES]

    # Make copy of training data to manipulate
    df = train.copy()
    df[TARGET] = target

    ### Numeric Features ###

    # Report current section of program
    print()
    print("=" * 10, "EVALUATE CORRELATION", "=" * 10)
    print()
    print("*" * 5, "Note", "*" * 5)
    print()
    print("Refer to 'thoughts.md' for interpretations of results.")
    print()
    print("*" * 5, "Numeric Features", "*" * 5)
    print()

    # Calculate pointbiserialr corr b/t num-cat (like Pearson's R for num-num)
    print("The r-values (-1, 1) are:")
    for feat in num:
        corr = pointbiserialr(x=target, y=train[feat])
        print(feat, ":", round(corr.statistic, 6))

    # Box plot of rate of "female" in each numeric feature
    # r-value will capture upward/downward trends, but not middle-ward like
    # "mid-length names tend to be female"
    fig, ax = plt.subplots(nrows=len(num), ncols=1, figsize=(10, 3 * len(num)))
    for i in range(len(num)):
        # Bar graph of a 5-binned feature with its count/rate of female names
        df["bin"] = pd.qcut(df[num[i]], 5, duplicates="drop")
        df.groupby("bin")[TARGET].mean().plot(kind="bar",
                                              ax=ax[i],
                                              xlabel=num[i],
                                              ylabel="Rate of 'Female'",
                                              ylim=(0, 1),
                                              rot=0,
                                              )
    fig.suptitle("Rate of 'Female' by Numeric Feature (Binned)")
    fig.tight_layout(h_pad=2)
    fig.savefig("female_rate.png")
    plt.close(fig)

    # Get heatmap of numeric and binary features to check for redundanct features
    corr = df[num + binary].corr()
    fig, ax = plt.subplots(figsize=(13, 8))
    sns.heatmap(corr, annot=True, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_title("Correlations among Numeric and Binary Features")
    fig.savefig("corr_heatmap.png")
    plt.close(fig)

    # Report rate of "female" in numeric features
    print()
    print("Successfully saved bar charts.")
    print("Successfully saved heatmap.")

    ### Categorical Features ###

    print()
    print("*" * 5, "Categorical Features", "*" * 5)
    print()

    # Calculate chi^2 corr b/t cat-cat (like Pearson's R for num-num)
    print("The Cramer's V (0-1) association values are:")
    for feat in cat:
        crosstab = pd.crosstab(df[feat], df[TARGET])
        # chi2 = chi2_contingency(crosstab)
        corr = association(crosstab, method="cramer")
        # print(feat, "p-value :", round(chi2.pvalue, 6))
        print(feat, ":", round(corr, 6))

    # Calculate rate of "female" across each categorical var
    for feat in cat:
        counts_crosstab = pd.crosstab(df[feat], df[TARGET])
        counts_df = pd.DataFrame()
        counts_df["male"] = counts_crosstab[0]
        counts_df["female"] = counts_crosstab[1]
        counts_df["total"] = counts_df["male"] + counts_df["female"]
        counts_df["rate_female"] = round(counts_df["female"] / counts_df["total"], 2)

        # Report rate of "female" in categorical vars
        print()
        print("Rate of female across ',", feat, "':", sep="")
        print(counts_df.sort_values(by="rate_female", ascending=False))

def build_pipeline(num: list, cat: list) -> Pipeline:
    """
    Build a pipeline given the numerical and categorical features.

    Parameters:
        num (list): numeric feature names.
        cat (list): categorical feature names.

    Returns:
        Pipeline.
    """

    letter = [feat for feat in cat if feat in LETTER_FEATURES]
    binary = [feat for feat in cat if feat not in LETTER_FEATURES]

    # Create feature transformers
    transformers = []
    if num:
        transformers.append(("num", StandardScaler(), num))
    if letter:
        # EX: Do not want to compare "a"-ness, so leave all 26 columns
        transformers.append(("letter", OneHotEncoder(categories=[ALPHABET] * len(letter),
                                                     handle_unknown="ignore"),
                                                     letter,
                                                     ))
    if binary:
        # EX: Want to compare binary results, so best drop one output
        transformers.append(("binary", OneHotEncoder(categories=[[0, 1]] * len(binary),
                                                     drop="if_binary"),
                                                     binary,
                                                     ))

    # Return pipeline
    return Pipeline([
        ("prep", ColumnTransformer(transformers)),
        ("lr", LogisticRegression(fit_intercept=True,   # add col of 1's
                                  C=1.0,                # strength of regularization
                                  l1_ratio=1,           # lasso
                                  solver="liblinear",   # fast optimizer for small data
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
            stats["range_"+metric_name] = max(values) - min(values)
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
# THE BASELINE AND THE PRIOR
###################################################################################

# NOTE: always predict 0.5 -> always log loss of 0.693147.

# # Baseline for any logistic regression is 0.5 probability
# BASELINE = 0.5

# # Evaluate baseline model
# evaluate_basic_model(model_name="BASELINE", prob=BASELINE, target=target)

# NOTE: turns out that there are an equal number of male and
# female names in the train set, so prior == baseline.

# # Calculate the prior: num_female / total
# prior = len(target[target == 1]) / len(target)

# # Evaluate the prior
# evaluate_basic_model(model_name="PRIOR", prob=prior, target=target)

###################################################################################
# EVALUATE CORRELATION AMONG FEATURES
###################################################################################

evaluate_correlation(num=NUM_FEATURES,
                     cat=CAT_FEATURES,
                     target=target,
                     train=train,
                     )

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
