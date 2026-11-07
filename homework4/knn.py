"""
Olivia Buchanan
Homework 4: KNN, Cross-validation, and PCA
"""

import spotifysongs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.linear_model import LinearRegression

###################################################################################
# CONSTANTS
###################################################################################

# Seed
SEED = 17
np.random.seed(SEED)

# Load all tables as their own variable
globals().update(spotifysongs.load())

print()
print("Successfully loaded in Spotify songs...")

# Cross validation settings
NUM_FOLDS = 5

# Targets
CAT_TARGET = "mode"
NUM_TARGET = "popularity"

# Extra info for graphs
LABEL = "title"

# Numeric features
NUM_FEATURES = [
    "tempo",
    "valence",
    "speechiness",
]

# Categorical features
TIME_SIGS = ["time_signature_3", "time_signature_4", "time_signature_5"]

###################################################################################
# FUNCTIONS
###################################################################################

def build_pipeline(model_type: str,
                   num: list,
                   cat: list,
                   passthrough: list,
                   k: int = 10,
                   n_comp: int = 2,
                   ) -> Pipeline:
    """
    Build a pipeline given the numerical and categorical features.

    Parameters:
        model_type (str): knn, pca, or pcr model.
        num (list): numeric feature names.
        cat (list): categorical feature names.
        passthrough (list): features to leave as-is.
        k (int): number of neighbors.
        n_comp (int): number of components.

    Returns:
        Pipeline.
    """

    # Create feature transformers
    transformers = []
    if num:
        transformers.append(("num", StandardScaler(), num))
    if cat:
        transformers.append(("cat", OneHotEncoder(sparse_output=False,
                                                  handle_unknown="ignore",
                                                  drop="first"),
                                                  cat,
                                                  ))
    if passthrough:
        transformers.append(("pass", "passthrough", passthrough))

    # Return Pipeline
    if model_type == "knn":
        return Pipeline([
            ("prep", ColumnTransformer(transformers)),
            ("knn", KNeighborsClassifier(n_neighbors=k)),
        ])
    elif model_type == "pca":
        return Pipeline([
            ("prep", ColumnTransformer(transformers)),
            ("pca", PCA(n_components=n_comp)),
        ])
    elif model_type == "pcr":
        return Pipeline([
            ("prep", ColumnTransformer(transformers)),
            ("pca", PCA(n_components=n_comp)),
            ("lr", LinearRegression(fit_intercept=True)),
        ])
    else:
        return None        

###################################################################################
# PART 1: DRAW KNN DECISION REGIONS
###################################################################################

print()
print("=" * 10, "PART 1: DECISION REGIONS", "=" * 10)

# Split target from features
cat_target = songs[CAT_TARGET]
features_df = songs[NUM_FEATURES[:2]]

# Set 3 k-values to explore
k_values = [1, 10, 50]

# Create plot demonstrating what happens as k-value changes
fig, ax = plt.subplots(nrows=len(k_values), ncols=1, figsize=(10, 3 * len(k_values)))

# Get color map
cmap = plt.get_cmap("viridis_r")

# Fill plot with decision regions
for i in range(len(k_values)):

    # Get KNN pipeline
    knn_pipeline = build_pipeline(model_type="knn",
                                  num=NUM_FEATURES[:2],
                                  cat=[],
                                  passthrough=[],
                                  k=k_values[i])

    # Fit pipeline
    knn_pipeline.fit(features_df, cat_target)

    # Get decision regions
    DecisionBoundaryDisplay.from_estimator(
        knn_pipeline,
        features_df,
        ax=ax[i],
        response_method="predict",
        plot_method="contourf",
        cmap=cmap,
        vmin=0,
        vmax=1,
    )

    # Fill plot with song data
    ax[i].scatter(
        features_df[NUM_FEATURES[0]], features_df[NUM_FEATURES[1]],
        c=(cat_target == knn_pipeline.classes_[1]),
        alpha=0.3,
        s=20,
        cmap=cmap,
    )

    # Axis title
    axis_title = str(k_values[i]) + "-Nearest-Neighbors"
    ax[i].set_title(axis_title)

# Plot title and save
fig.legend(handles=[
        Patch(color=cmap(0.0), label="minor"),
        Patch(color=cmap(1.0), label="major"),
    ])
fig.suptitle("Decision Regions as k-value Changes")
fig.tight_layout(h_pad=2)
fig.savefig("decision_regions.png")
plt.close(fig)

print()
print("Successfully saved 'decision_regions.png'...")

###################################################################################
# PART 2: FIND GOOD K-VALUE
###################################################################################

print()
print("=" * 10, "PART 2: FIND GOOD K-VALUE", "=" * 10)
print()

# Split target from features
cat_target = songs[CAT_TARGET]
features_df = songs[NUM_FEATURES + TIME_SIGS]

# Get k-values
k_values = range(1, 76)

# Get KNN pipeline
knn_pipeline = build_pipeline(model_type="knn",
                              num=NUM_FEATURES,
                              cat=[],
                              passthrough=TIME_SIGS,
                              )

# Make GridSearchCV for exploring k-values
# Note: any GridSearch settings override underlying pipeline
grid = GridSearchCV(
    knn_pipeline,
    param_grid={"knn__n_neighbors": k_values},
    cv=KFold(NUM_FOLDS, shuffle=True, random_state=SEED),
    scoring="accuracy",
    n_jobs=-1,
)

# Fit GridSearchCV
grid.fit(features_df, cat_target)

# Print GridSearchCV results
print("Range of k-values:", k_values)
print("Best k-value:", grid.best_params_["knn__n_neighbors"])
print("Best CV accuracy: {score:.5f}".format(score=grid.best_score_))

# Print results of baseline majority guess
major_ratio = len(cat_target[cat_target == 1]) / len(cat_target)
print()
print("Baseline majority guess prediction:", "major" if major_ratio > 0.5 else "minor")
print("Baseline majority guess accuracy:", round(max(major_ratio, 1 - major_ratio), 5))

# Make graph of accuracy as k-value changes
fig, ax = plt.subplots()
n_neighbors = grid.cv_results_["param_knn__n_neighbors"]
accuracies = grid.cv_results_["mean_test_score"]
ax.plot(n_neighbors, accuracies, marker="o", color="green")
ax.set_xlabel("Number of Neighbors")
ax.set_ylabel("Mean cross-validation accuracy")
fig.suptitle("Accuracy as k-value Changes")
fig.tight_layout(h_pad=2)
fig.savefig("knn_accuracy.png")
plt.close(fig)

print()
print("Successfully saved 'knn_accuracy.png'...")
print()
