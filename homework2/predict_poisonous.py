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
from sklearn.metrics import accuracy_score, f1_score, log_loss

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
SCORING = ["neg_log_loss"] # also accuracy and f1
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
FINAL_FEATURES = {"num": ["cap_diameter", "stem_width"],
                  "cat": ["cap_shape", "cap_color", "gill_color", "habitat", "season"],
                  }

# Models to evaluate
SCHEMAS = [
        # all numeric, all cat, all features
        {"name": "ALL NUMERIC",
         "num": NUM_FEATURES,
         "cat": [],
         },
        {"name": "ALL CATEGORICAL",
         "num": [],
         "cat": CAT_FEATURES,
         },
        {"name": "ALL FEATURES",
         "num": ["cap_diameter", "stem_height", "stem_width"],
         "cat": ["cap_shape", "cap_color", "does_bruise_or_bleed", "gill_color", "habitat", "season"]
         },
        {"name": "FORWARD SELECTION",
         "num": FINAL_FEATURES["num"],
         "cat": FINAL_FEATURES["cat"],
        },
        ]

#####################################################
# FUNCTIONS
#####################################################

def evaluate_basic_model(prob: float, target: pd.Series) -> dict:
    """
    Evaluate the a basic model that predicts the same thing.

    Parameters:
        prob (float): the probability.
        target (pd.Series): the target.

    Returns:
        Dictionary: accuracy, f1, neg_log_loss.
    """

    # Get list of probabilities and predictions
    probs = [prob for _ in range(len(target))]
    preds = [1 if prob > 0.5 else 0 for _ in range(len(target))]
    
    # Return results
    return {"accuracy": accuracy_score(target, preds),
            "f1": f1_score(target, preds),
            "neg_log_loss": -log_loss(target, probs),
            }

def print_basic_model(model_name: str,
                      prob: float,
                      results: dict,
                      ) -> None:
    """
    Prints the results from the basic model.
    
    Parameters:
        model_name (str): name of the model.
        prob (float): the constant probability.
        results (dict): accuracy, f1, neg_log_loss.

    Returns:
        None. Print statements.
    """

    # Print settings
    print()
    print("=" * 10, model_name, "=" * 10)
    print()
    print("*PREDICTION*")
    print("Probability:", prob)
    print("Always predict:", 1 if prob > 0.5 else 0)
    
    # Print results
    print()
    print("*RESULTS*")
    print("accuracy:", round(results["accuracy"], 6))
    print("f1:", round(results["f1"], 6))
    print("neg_log_loss:", round(results["neg_log_loss"], 6))

def build_pipeline(num: list, cat: list) -> Pipeline:
    """
    Build a pipeline given the numerical and categorical features.

    Parameters:
        num (list): numeric feature names.
        cat (list): categorical feature names.

    Returns:
        Pipeline.
    """

    # Create feature transformers
    transformers = []
    if num:
        transformers.append(("num", StandardScaler(), num))
    if cat:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), cat))

    # Return pipeline
    return Pipeline([
        ("prep", ColumnTransformer(transformers)),
        ("lr", LogisticRegression(fit_intercept=True)),
    ])

def evaluate_cross_validation(model: dict,
                              target: pd.Series,
                              train: pd.DataFrame,
                              ) -> dict:
    """
    Evaluate logistic regression on the dictionary of models.

    Parameters:
        model (dict): schema of model names and features.
        target (pd.Series): the target all models aim for.
        train (pd.DataFrame): the features.

    Returns:
        Dict. Returns model results as a dictionary.
    """

    # Create Pipeline object
    model_pipeline = build_pipeline(num=model["num"],
                                    cat=model["cat"],
                                    )

    # Get list of features
    features = model["num"] + model["cat"]

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
        if "test_" in metric_name:
            results[metric_name] = values

    return results

def do_forward_selection(target: pd.Series,
                         all_train_features: pd.DataFrame,
                         prior_log_loss: float,
                         tolerance: float = 0.001,
                         ) -> tuple:
    """
    Perform forward selection for all given features against the target.

    Parameters:
        target (pd.Series): the target.
        all_train_features (pd.DataFrame): all the features.
        prior_log_loss (float): initial log loss to beat.
        tolerance (float): when to stop forward selection.
    
    Returns:
        Tuple: (selected features, history).
    """

    print()
    print("=" * 10, "FORWARD SELECTION PROCESS", "=" * 10)

    # Track selected features
    selected = []
    candidate_features = NUM_FEATURES + CAT_FEATURES

    # Track best feature and log losses
    overall_best_log_loss = prior_log_loss

    # Return history of features
    history = []

    # Begin adding candidate features one round at a time
    done = False
    counter = 1
    while (not done) and (candidate_features):

        # Track the best log loss and feature for a round
        round_best_log_loss = -10
        round_best_feature = ""

        # Loop through candidate features
        for candidate in candidate_features:

            # Get features for a model
            current_features = selected + [candidate]
            model = {"num": [feat for feat in current_features if feat in NUM_FEATURES],
                     "cat": [feat for feat in current_features if feat in CAT_FEATURES],
                     }

            # Evaluate whether average log loss of model beat the round's best
            scores = evaluate_cross_validation(model=model,
                                               target=target,
                                               train=all_train_features,
                                               )
            avg_log_loss = np.average(scores["test_neg_log_loss"])
            if avg_log_loss > round_best_log_loss:
                round_best_log_loss = avg_log_loss
                round_best_feature = candidate

        # Evaluate whether the round's best log loss beat the overall log loss
        difference = round_best_log_loss - overall_best_log_loss
        if difference < tolerance:
            done = True
        else:
            overall_best_log_loss = round_best_log_loss
            selected.append(round_best_feature)
            candidate_features.remove(round_best_feature)
            history.append({"feature_added": round_best_feature, "neg_log_loss": overall_best_log_loss})
            print()
            print("Round ", counter)
            print(history[-1])

        # Update the counter
        counter += 1

    return (selected, history)

# def print_evaluation_results(models: list,
#                              results: list,
#                              ) -> None:
#     """
#     Print evaluation results neatly.

#     Parameters:
#         models (list): schema of model names and features.
#         results (list): parallel list of model results.

#     Returns:
#         None. Print statements.
#     """

#     for model, scores in zip(models, results):
#         # Print model name and features
#         print()
#         print("=" * 10, "LOGISTIC REGRESSION:", model["name"], "=" * 10)
#         print()
#         print("*FEATURES*")
#         print(model["num"] + model["cat"])
#         print()
#         print("*RESULTS (greater is better)*")

#         # Print results
#         print()
#         for metric_name, values in scores.items():
#             print()
#             print("average "+metric_name+":", round(np.average(values), 6))
#             if "neg_log_loss" in metric_name:
#                 print("range test_neg_log_loss:", round([max(values) - min(values), 6]))

def print_worst_results(models: list, results: list) -> None:
    """
    Print the models with the best performances.

    Parameters:
        models (list): schema of model names and features.
        results (list): parallel list of model results.
    
    Returns:
        None. Print statements.
    """

    # Find max scores and log loss spread of each model
    model_stats = []
    for model in results:
        stats = {}
        for metric_name, values in model.items():
            stats[metric_name] = np.average(values)
            if "neg_log_loss" in metric_name:
                stats["range test_neg_log_loss"] = max(values) - min(values)
        model_stats.append(stats)

    # Get maxes as a df
    stats_df = pd.DataFrame(model_stats, index=[model["name"] for model in models])

    # Print results
    print()
    print("=" * 10, "WORST LOG LOSS", "=" * 10)
    print()
    for metric_name in stats_df.columns:
        print()
        if "range" in metric_name:
            print("Worst "+metric_name+":", round(stats_df[metric_name].max(), 6))
        else:
            print("Worst "+metric_name+":", round(stats_df[metric_name].min(), 6))
        print("Model:", stats_df[metric_name].idxmin())
    print()
    print(stats_df)

def get_betas(model_pipeline: Pipeline) -> tuple:
    """
    Get intercept and coefficients of a model.

    Parameters:
        model_pipeline: Pipeline for a model.

    Returns:
        Tuple: (intercept, coefs)
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

#####################################################
# LOAD DATA
#####################################################

# Load datacoefficients
train = pd.read_csv(TRAIN_FILENAME)
test = pd.read_csv(TEST_FILENAME)

# Get target and all features
target = train[TARGET]
all_train_features = train[NUM_FEATURES + CAT_FEATURES]

# Check scoring metrics available in CV
# print(get_scorer_names())

#####################################################
# THE BASELINE: PREDICT THE BASELINE
#####################################################

# Baseline for any logistic regression is 0.5 probability
BASELINE = 0.5

# Evaluate baseline model
baseline_results = evaluate_basic_model(prob=BASELINE, target=target)

# Print results for baseline model
print_basic_model(model_name="THE BASELINE",
                  prob=BASELINE,
                  results=baseline_results,
                  )

#####################################################
# THE PRIOR: PREDICT MAJORITY CLASS
#####################################################

# Calculate the prior: num_poisonous / total
prior = len(target[target == 1]) / len(target)

# Evaluate the prior
prior_results = evaluate_basic_model(prob=prior, target=target)

# Print results for the prior
print_basic_model(model_name="THE PRIOR",
                  prob=prior,
                  results=prior_results,
                  )

#####################################################
# DO LOGISTIC REGRESSION WITH CROSS VALIDATION
#####################################################

# Do forward feature selection
best_features, history = do_forward_selection(target=target,
                                              all_train_features=all_train_features,
                                              prior_log_loss=prior_results["neg_log_loss"],
                                              )

print()
print("Best features:")
print(best_features)

# Fill parallel list of model scores
results = []
for model in SCHEMAS:
    scores = evaluate_cross_validation(model=model,
                                       target=target,
                                       train=all_train_features,
                                       )
    results.append(scores)

# Print all of the results
# print_evaluation_results(models=SCHEMAS,
#                          results=results,
#                          )

# Print best models in the results
print_worst_results(models=SCHEMAS,
                    results=results,
                    )

#####################################################
# PREDICT ON TEST SET AND SUBMIT
#####################################################

# Create final model pipeline object
model_pipeline = build_pipeline(num=FINAL_FEATURES["num"],
                                cat=FINAL_FEATURES["cat"],
                                )

# Combine FINAL_FEATURES into a list
list_final_features = FINAL_FEATURES["num"] + FINAL_FEATURES["cat"]

# Actually train the model on training data
model_pipeline.fit(train[list_final_features], target)

# Scale and encode test data, predict probabilities on test data
p_poisonous = model_pipeline.predict_proba(test[list_final_features])[:, 1]

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

# Get the coefficients of the model
intercept, coefs = get_betas(model_pipeline=model_pipeline)

# Print final features picked
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
print()
