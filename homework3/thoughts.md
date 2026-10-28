# Thoughts

This file interprets the png's generated from
evaluate_correlation().

## Correlation of Numeric Features

**p-value**
* Not calculated.
* With a sample size of 5000, small differences will be deemed
significant oftentimes.

**r-value**
* How: calculate r-value with pointbiserial r because using num-cat types.
* Interpret like Pearson's r:
    * Magnitude indiciates strength of correlation with target
    * Sign indicates relationship of correlation with target
* My results:
    * Each numeric feature has weak, positive correlation with the target. 
    * "vowel_ratio" has highest correlation at 0.300 so far.

**female_rate.png**
* How: bin each numeric feature into 5 slots, then find rate of female for
that bin.
* How to interpret:
    * r-value only captures a downward or upward trend of female-ness.
    * This bar graph checks for "mid-length names tend to be female"
    if there is a hump in the middle of the trend.
* My results:
    * Upward trend for features.
    * r-value didn't miss anything.

**corr_heatmap.png**
* How: correlation of numeric and binary features with each other, not
wit the target.
* How to interpret: just like Pearson's r.
* My results:
    * "ends_n" and "ends_vowel" are negatively correlated (makes sense
    since they are mutually exclusive).
    * "num_vowels" is strongly, positively correlated with "length" and
    "vowel_ratio", likely because you can calculate it with them.
    * "length" and "vowel_ratio" are weakly correlated with each other.
    * "num_vowels" seems to not add new info. Just use "length" and
    "vowel_ratio."

## Correlation of Categorical Features

* p-values:
    * All are pretty much 0, so reject null -> yes correlation.
    * As mentioned, this happens because it's easy to pick up differences
    in 5,000 observations.
    * Negligible. Commented out. Look at association instead.
* Cramer's V:
    * What: measures strength of association of cat-cat on 0-1 scale.
    * How: bigger value, stronger association.
    * Interpretation: keep "last_letter" over "ends_n" and "ends_vowel"
    because they are redundant by construction; keep first_letter because
    it has markedly strong association compared to ends_n but has different
    info.
* Rate of "female" by "last_letter" measures direction of association:
    * Which letter tends to be "female"?
    * Some "rate_female" are high because the total is high.
    * Some "rate_female" are 0.2 with high totals, which indicates male-ness.
    * Some "rate_female" are 0, but they come from a negligible, small pool of males.

