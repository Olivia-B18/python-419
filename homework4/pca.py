"""
Olivia Buchanan
Homework 4: KNN, Cross-validation, and PCA
"""

import spotifysongs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
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
# PART 3: DISCOVER PRINCIPAL COMPONENTS
###################################################################################

print()
print("=" * 10, "PART 3: DISCOVER PRINCIPAL COMPONENTS", "=" * 10)

# Get features
features_df = songs[NUM_FEATURES + TIME_SIGS]

# Get PCA pipeline
n_comp = 3
pca_pipeline = build_pipeline(model_type="pca",
                              num=NUM_FEATURES,
                              cat=[],
                              passthrough=TIME_SIGS,
                              n_comp=n_comp,
                              )

# Fit PCA
pca_pipeline.fit(features_df)

# Get loadings from PCA
loadings = pd.DataFrame(
    pca_pipeline.named_steps["pca"].components_.T,
    columns=[f"PC{n}" for n in range(1, n_comp + 1)],
)
loadings.insert(0, "feature", NUM_FEATURES + TIME_SIGS)

# Display loadings
print()
print("The loadings:")
print(loadings)

# Get and display explained variance ratios
var_ratios = pca_pipeline.named_steps["pca"].explained_variance_ratio_
print()
print("Proportions of variance explained:")
for i in range(len(var_ratios)):
    print("PC{n_comp}: {ratio}".format(n_comp= i + 1,
                                       ratio=var_ratios[i],
                                       ))

# Get and display cumulative explained variance ratios
cum_var_ratios = np.cumsum(var_ratios)
print()
print("Total variance explained:")
for i in range(len(cum_var_ratios)):
    print("Combined PCs up to PC{n_comp}: {ratio}".format(n_comp= i + 1,
                                                          ratio=cum_var_ratios[i],
                                                          ))

"""
PCA interpretations (revised with Claude):

Three components capture 94.3% of the variation in
these six features (3 numeric, 1 categorical with 3 classes).

Time signature loadings are low because OneHotEncoding
left them unscaled. This will result in them having low
variance relative to scaled numeric features since the
only values present are 0/1.

Since each of the 3 components' explained variation are
similar in size, that means that the numeric features are
nearly uncorrelated, so there isn't much compression that
can be done.

PC1 is informed mostly by tempo and valence with negative
speechiness (or spoken words). It represents upbeat vibes
of a song, such as those with more instrumental parts or
singing.

PC2 is informed by speechiness, valence, and no tempo. It
represents songs with lots of talking and upbeat vibes.

PC3 is informed by tempo, speechiness, and negative valence.
It seems to represent fast songs with lots of talking and do
not give positive vibes. Perhaps genres like rap.

Interesting how each component is just a combination of tempo,
valence, and speechiness, but time signature plays a minimal
role due to lack of scaling. This suggests that most of my
information/variance in my components from numeric features.
"""

###################################################################################
# PART 4: MAP INTO PC SPACE
###################################################################################

print()
print("=" * 10, "PART 4: MAP INTO PC SPACE", "=" * 10)

# Get target and features
cat_target = songs[CAT_TARGET]
features_df = songs[NUM_FEATURES + TIME_SIGS]

# Get PCA pipeline
n_comp = 2
pca_pipeline = build_pipeline(model_type="pca",
                              num=NUM_FEATURES,
                              cat=[],
                              passthrough=TIME_SIGS,
                              n_comp=n_comp,
                              )

# Fit and transform PCA with only 2 components
pca12 = pca_pipeline.fit_transform(features_df)

# Create scatterplot using PC1 and PC2 as axes
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 5),
                       sharex=True, sharey=True)
ax[0].scatter( # Major songs from PC1 and PC2
    pca12[cat_target == 1, 0],
    pca12[cat_target == 1, 1],
    marker=".",
    color="magenta",
    label="Major",
    alpha=0.3,
    s=20,
)
ax[1].scatter( # Minor songs from PC1 and PC2
    pca12[cat_target == 0, 0],
    pca12[cat_target == 0, 1],
    marker=".",
    color="orange",
    label="Minor",
    alpha=0.3,
    s=20,
)

# Songs to label: (artist, title)
LABELED_SONGS = [
    # Major
    ("Taylor Swift", "august"),
    ("Ariana Grande", "positions"),
    ("Marshmello", "Happier"),
    # Minor
    ("Why Don't We", "8 Letters"),
    ("Chris Brown", "Under The Influence"),
    ("Sufjan Stevens", "To Be Alone With You"),
]

# Annotate points on the scatter plot with the song title
# The following for loop was solely generated by Claude Code
for artist, title in LABELED_SONGS:

    # Find the song's row number
    match = songs["artists"].str.contains(artist, regex=False) & (songs["title"] == title)
    row = np.flatnonzero(match)[0]

    # Get its coordinates and choose the matching panel
    x, y = pca12[row]
    panel = ax[0] if cat_target.iloc[row] == 1 else ax[1]

    # Highlight and label the point
    panel.scatter(x, y, color="black", s=40, zorder=3)
    panel.annotate(title, (x, y), xytext=(6, 6), textcoords="offset points",
                   fontsize=9, fontweight="bold", zorder=4, rotation=25)

# Finishing touches to the plot
fig.supxlabel("Principal Component 1 ('Upbeat Sounds')")
fig.supylabel("Principal Component 2 ('Upbeat with Talking')")
fig.legend()
fig.suptitle("Scatterplot of PC1 and PC2 on Song Data")
fig.tight_layout(h_pad=2)
fig.savefig("PC12.png")
plt.close(fig)

print()
print("Successfully saved 'PC12.png'...")

"""
Observations:

Two clusters in each subplot for major and minor,
but one is more dominant than the other.

Most songs lie in the dominant cluster, indicating
that they are very upbeat but don't have a lot of talking.

A handful of songs lie in the lesser cluster, indicating
that they are not as upbeat but do have more talking.
"""

###################################################################################
# PART 5: PCR FOR NUMERIC TARGET
###################################################################################

print()
print("=" * 10, "PART 5: PCR FOR NUMERIC TARGET", "=" * 10)

# Get target and features
num_target = songs[NUM_TARGET]
features_df = songs[NUM_FEATURES + TIME_SIGS]

# Get PCR pipeline
pcr_pipeline = build_pipeline(model_type="pcr",
                              num=NUM_FEATURES,
                              cat=[],
                              passthrough=TIME_SIGS,
                              )

# Get n_comp values to tune on
n_comp_values = range(1, len(features_df.columns) + 1)

# Make GridSearchCV for exploring k-values
# Note: any GridSearch settings override underlying pipeline
grid = GridSearchCV(
    pcr_pipeline,
    param_grid={"pca__n_components": n_comp_values},
    cv=KFold(NUM_FOLDS, shuffle=True, random_state=SEED),
    scoring="r2",
    n_jobs=-1,
)

# Fit GridSearchCV
grid.fit(features_df, num_target)

# Print GridSearchCV results
print()
print("Range of k-values:", n_comp_values)
print("Best n_comp:", grid.best_params_["pca__n_components"])
print("Best CV R^2: {score:.5f}".format(score=grid.best_score_))

# Make graph of R^2 as n_comp changes
fig, ax = plt.subplots()
n_comps = grid.cv_results_["param_pca__n_components"]
r2_scores = grid.cv_results_["mean_test_score"]
ax.plot(n_comps, r2_scores, marker="o", color="purple")
ax.set_xlabel("Number of Components")
ax.set_ylabel("Mean cross-validation R^2")
fig.suptitle("R^2 as Number of Components Changes")
fig.tight_layout(h_pad=2)
fig.savefig("pcr_r2.png")
plt.close(fig)

print()
print("Successfully saved 'pcr_r2.png'...")
print()
