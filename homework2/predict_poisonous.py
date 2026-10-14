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

SEED = 17
TRAIN_FILENAME = "train.csv"
TEST_FILENAME = "test.csv"
SUBMISSION_FILENAME = "submission.csv"
SCORING = ["accuracy", "f1", "neg_log_loss"]
NUM_CV = 10
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
CHOSEN_FEATURES = NUM_FEATURES.copy()

#####################################################
# FUNCTIONS
#####################################################

def evaluate_logistic_regression(preprocess: ColumnTransformer,
                                 model_name: str):
    """
    Take processed features and evaluate logistic regression.

    Parameters:
        preprocess (ColumnTransformer): features to be scaled or encoded according to column name.
        model_name (str): quick name to describe features of model.

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
            )

    # Print settings and results
    print()
    print("=" * 10, "LOGISTIC REGRESSION:", model_name, "=" * 10)
    print()
    print("*SETTINGS*")
    print("Num CV folds:", NUM_CV)
    print("Features:", CHOSEN_FEATURES)
    print()
    print("*RESULTS (closer to 1 is better)*")
    for metric in SCORING:
        print("test_"+metric+":", np.average(results["test_"+metric]))


#####################################################
# LOAD DATA
#####################################################

# Load data
train = pd.read_csv(TRAIN_FILENAME)
test = pd.read_csv(TEST_FILENAME)

# Separate features and target
target = train[TARGET]
features = train[CHOSEN_FEATURES]

# Check scoring metrics available in CV
# print(get_scorer_names())

#####################################################
# THE PRIOR
#####################################################

# Note: no train and validate split needed since
# this model only uses one parameter.

# Calculate the prior: num_poisonous / total
prior = len(target[target == 1]) / len(target)

# Get list of probabilities and predictions
prior_probs = [prior for _ in range(len(target))]
prior_preds = [1 if prior > 0.5 else 0 for _ in range(len(target))]

# Evaluate the prior
prior_accuracy = accuracy_score(target, prior_preds)
prior_f1_score = f1_score(target, prior_preds)
prior_log_loss = log_loss(target, prior_probs)

# Print the settings and results
print()
print("=" * 10, "MODEL: THE PRIOR", "=" * 10)
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

# Preprocess features 
prep_num_features = ColumnTransformer([
    ("num", StandardScaler(), NUM_FEATURES),
    #("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
    ])

# Logistic Regression models
evaluate_logistic_regression(preprocess=prep_num_features,
                             model_name="ALL NUMERIC",
                             )

print()

#####################################################
# PREDICT ON TEST SET AND SUBMIT
#####################################################

# TODO: update ColumnTransformer with chosen features

# # Create final model pipeline object
# prep_chosen_features = ColumnTransformer([
#     ("num", StandardScaler(), NUM_FEATURES),
#     ("cat", OneHotEncoder(handle_unkown="ignore"), CAT_FEATURES),
#     ])
# model_pipeline = Pipeline([
#     ("prep", prep_chosen_features),
#     ("lr", LogisticRegression(fit_intercept=True)),
#     ])
# 
# # Actually train the model on training data
# model_pipeline.fit(features, target)
# 
# # Scale and encode test data, predict probabilities on test data
# p_poisonous = model_pipeline.predict_proba(test[CHOSEN_FEATURES])[:, 1]
# 
# # Create df for submission
# submission = pd.DataFrame({
#     "id": test["id"],
#     "poisonous": p_poisonous,
# })
# 
# # Assert the df is valid
# assert len(submission) == len(test), "Submission does not have predictions for all rows in test data."
# assert list(submission.columns) == ["id", "poisonous"], "Submission columns must be only 'id' and 'poisonous'."
# assert submission["id"].is_unique, "Submission id's must be unique."
# assert submission["poisonous"].max() <= 1, "Submission probabilities must not exceed 1."
# assert submission["poisonous"].min() >= 0, "Submission probabilities must not fall below 0."
# 
# # Write submission to csv
# submission.to_csv(SUBMISSION_FILENAME, index=False)
# 
# # Read csv to check that index is missing
# written = pd.read_csv(SUBMISSION_FILENAME)
# assert list(written.columns) == ["id", "poisonous"], "Submission file must not include the pandas index."
# 
#####################################################
# END
#####################################################

# Print end message
# print()
# print("=" * 10, "END PROGRAM", "=" * 10)
# print()
# print("Wrote predictions to:", SUBMISSION_FILENAME)
# print()

