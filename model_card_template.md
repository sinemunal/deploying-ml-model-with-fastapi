# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Sinem Unal created the model for the project. 
There is only one model used where a simple hyper-parameter tuning was performed. RandomForestClassifier from scikit-learn package (version 1.2.2) is used.
Hyperparameters as a result of simple grid search CV: `n_estimators:100`, `max_depth:3` and the rest is default values.


## Intended Use
This model can be used to determine whether a certain person in a population with given features 
can make more than 50K for a year or not. It could be used by researchers to understand how the income would look like in past.

## Training Data
census dataset from <a href="https://archive.ics.uci.edu/ml/datasets/census+income" target="_blank">here</a> is used where random split with stratification is applied.
80% is used as training (further division of train dataset into 3 fold validation for hyperparameter tuning).
## Evaluation Data
20% of the whole dataset is used as test dataset.

## Metrics
Precision, recall and fbeta are used to determine model performance. Results on the test data are 0.47883787661406024, 0.8514030612244898, 0.6129476584022039 respectively.


## Ethical Considerations
When we look at the performance metrics calculated on female and male separately, we see that there is some discrepancy on the precision and recall for 
male and female as can be seen below.

Metrics | precision | recall | fbeta  
--- | --- |--------|----
Male | 0.461128 | 0.914588    | 0.613124
Female|  0.762195 |  0.510204 | 0.611247
If there is a predetermined group fairness criteria (https://en.wikipedia.org/wiki/Fairness_(machine_learning)#Group_Fairness_criteria) then one should
also calculate the scores with respect to those measures and see if there is
any significant discrepancy on those metrics. 

## Caveats and Recommendations

Hyperparameter tuning is performed on a very high level. Larger hyperparameter search space can be used to improve the performance.
Likewise, feature engineering can be applied for further improvements.
