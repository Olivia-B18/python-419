"""
Olivia Buchanan
Homework 2: Mushroom Madness
"""

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, KFold
from sklearn.metrics import get_scorer_names, accuracy_score, f1_score, log_loss 

#####################################################
# CONSTANTS
#####################################################

# Seed
SEED = 17

# File names
TRAIN_FILENAME = "train.csv"
TEST_FILENAME = "test.csv"
SUBMISSION_FILENAME = "submission.csv"

# Cross validation settings
SCORING = ["accuracy", "f1", "neg_log_loss"]
NUM_CV = 10

# Column names for features
TARGET = "poisonous"
NUM_FEATURES = ["cap_diameter",
                "stem_height",
                "stem_width",
                ]
CAT_FEATURES = ["cap_shape",
                "cap_color",
                "does_bruise_or_bleed",
                "gill_color",
                "habitat",
                "season",
                ]
# FINAL_FEATURES = ?

#####################################################
# FUNCTIONS
#####################################################

def validate_submission(submission: pd.DataFrame,
                        len_test: int,
                        ) -> True:
    """
    Assert statements verify that submission is valid.

    Parameters:
        submission (pd.DataFrame): the DataFrame of mushroom
            id's and probabilities to submit.
        len_test (int): the length of the test data.

    Returns:
        True if the submission is valid. Else, assert error.
    """
    
    # Assert the df is valid
    assert len(submission) == len_test, "Submission does not have predictions for all rows in test data."
    assert list(submission.columns) == ["id", "poisonous"], "Submission columns must be only 'id' and 'poisonous'."
    assert submission["id"].is_unique, "Submission id's must be unique."
    assert submission["poisonous"].max() <= 1, "Submission probabilities must not exceed 1."
    assert submission["poisonous"].min() >= 0, "Submission probabilities must not fall below 0."

    # Write submission to csv
    submission.to_csv(SUBMISSION_FILENAME, index=False)

    # Read csv to check that index is missing
    written = pd.read_csv(SUBMISSION_FILENAME)
    assert list(written.columns) == ["id", "poisonous"], "Submission file must not include the pandas index."

    return True

def evaluate_model(probs: np.ndarray,
                   target: pd.Series,
                   ) -> tuple:
    """
    Evaluate accuracy, F1, and cross entropy loss of a model.
    
    Parameters:
        probs (list): prediction probabilities of the model.
        target (pd.Series): the true class labels.

    Returns:
        Tuple: accuracy, F1 score, negative log loss
    """
    
    # Get class predictions from the probabilities
    preds = [1 if prior > 0.5 else 0 for _ in range(len(target))]

    # Return accuracy, f1_score, and neg_log_loss
    return (accuracy_score(target, preds),
            f1_score(target, preds),
            log_loss(target, probs),
            )
    
def evaluate_cross_validation(preprocess: ColumnTransformer,
                                 model_name: str,
                                 chosen_feature_names: list,
                                 target: pd.Series,
                                 features: pd.DataFrame,
                                 ) -> None:
    """
    Take preprocessed features and evaluate logistic regression.

    Parameters:
        preprocess (ColumnTransformer): features to be scaled
            or encoded according to column name.
        model_name (str): short name to describe the model.
        chosen_feature_names (list): the names of the features.
        target (pd.Series): the target.
        features (pd.DataFrame): the features as a DataFrame.

    Returns:
        None. Print statements of evaluation.
    """

    # Create Pipeline object
    model_pipeline = Pipeline([
        ("prep", preprocess),
        ("lr", LogisticRegression(fit_intercept=True)),
        ])

    # Do cross validation
    results = cross_validate(
            model_pipeline,
            features,
            target,
            cv=KFold(NUM_CV, shuffle=True, random_state=SEED),
            scoring=SCORING,
            return_estimator=True,
            )

    # Print settings
    print()
    print("=" * 10, "LOGISTIC REGRESSION:", model_name, "=" * 10)
    print()
    print("*SETTINGS*")
    print("Num CV folds:", NUM_CV)
    print("Features:", chosen_feature_names)

    # Print results
    print()
    print("*RESULTS (greater is better)*")
    for metric in SCORING:
        print()
        print("min test_"+metric+":", round(min(results["test_"+metric]), 6))
        print("max test_"+metric+":", round(max(results["test_"+metric]), 6))
        print("average test_"+metric+":", round(np.average(results["test_"+metric]), 6))
        #print("test_"+metric+":", list(np.around(results["test_"+metric], 6)))

#####################################################
# LOAD DATA
#####################################################

# Load data
train = pd.read_csv(TRAIN_FILENAME)
test = pd.read_csv(TEST_FILENAME)

# Get target and all features
target = train[TARGET]
all_train_features = train.drop(columns=TARGET)

# Check scoring metrics available in CV
# print(get_scorer_names())

#####################################################
# THE BASELINE: PREDICT THE BASELINE
#####################################################

print()
print("=" * 10, "THE BASELINE", "=" * 10)
print()
print("Baseline:", 0.5)
print()
print("neg_log_loss:", log_loss(target, [0.5 for _ in range(len(target))]))

#####################################################
# THE PRIOR: PREDICT MAJORITY CLASS
#####################################################

# Note: no train and validate split needed since
# this model only uses one parameter.

# Calculate the prior: num_poisonous / total
prior = len(target[target == 1]) / len(target)

# Get list of probabilities 
prior_probs = [prior for _ in range(len(target))]

# Evaluate the prior
prior_accuracy, prior_f1_score, prior_log_loss = \
        evaluate_model(probs=prior_probs,
                       target=target,
                       )

# Print the settings and results
print()
print("=" * 10, "THE PRIOR", "=" * 10)
print()
print("*SETTINGS*")
print("Prior: num_poisonous / total = {prior:.5f}".format(prior=prior))
print()
print("*RESULTS (closer to 1 is better)*")
print("accuracy:", prior_accuracy)
print("f1_score:", prior_f1_score)
print("neg_log_loss:", -prior_log_loss)

# Consider what happens when you always predict 1 like prior_preds do...

#####################################################
# DO LOGISTIC REGRESSION WITH CROSS VALIDATION
#####################################################

# Note: no train and validate split needed since
# this evaluation uses cross validation.

# Model: all numeric
prep_num_features = ColumnTransformer([
    ("num", StandardScaler(), NUM_FEATURES),
    ])
evaluate_cross_validation(preprocess=prep_num_features,
                             model_name="ALL NUMERIC",
                             chosen_feature_names=NUM_FEATURES,
                             target=target,
                             features=all_train_features[NUM_FEATURES],
                             )

# Model: all categorical
prep_cat_features = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), CAT_FEATURES),
    ])
evaluate_cross_validation(preprocess=prep_cat_features,
                             model_name="ALL CATEGORICAL",
                             chosen_feature_names=CAT_FEATURES,
                             target=target,
                             features=all_train_features[CAT_FEATURES],
                             )

#####################################################
# PREDICT ON TEST SET AND SUBMIT
#####################################################

# TODO:
# Fill FINAL_FEATURES
# FIll prep_final_features
# Replace NUM_FEATURES with FINAL_FEATURES

# Create final model pipeline object
prep_final_features = ColumnTransformer([
    ("num", StandardScaler(), NUM_FEATURES),
    ])

model_pipeline = Pipeline([
    ("prep", prep_final_features),
    ("lr", LogisticRegression(fit_intercept=True)),
    ])

# Actually train the model on training data
model_pipeline.fit(train[NUM_FEATURES], target)

# Scale and encode test data, predict probabilities on test data
p_poisonous = model_pipeline.predict_proba(test[NUM_FEATURES])[:, 1]

# Create df for submission
predictions = pd.DataFrame({
    "id": test["id"],
    "poisonous": p_poisonous,
})

# Assert the df is valid
if validate_submission(submission=predictions, len_test=len(test)):
    print()
    print("=" * 10, "END PROGRAM", "=" * 10)
    print()
    print("Successfully wrote predictions to:", SUBMISSION_FILENAME)
    print()

