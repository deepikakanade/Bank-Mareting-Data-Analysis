# Bank-Mareting-Data-Analysis

## Requirements
* Python 2.7 
* Numpy >= 1.14.2
* Matplotlib >= 2.2.0
* Pandas >= 0.22.0
* Scikit-Learn >= 0.19.1

The data was collected as a marketing campaign to predict if a customer would make a term deposit in the bank.

The dataset considered for the project is 10% of the UCI bank Marketing dataset available online. The dataset has 4119 rows with 19 features.

The issues in the dataset were as follows:
-> The features had missing values which had to be imputed.
-> Preprocessing involved handling categorical data.
-> The dataset was imbalanaced. Number of class 1 (yes) labels were low compared to number of class 0 (no) labels.

## Preprocessing
Preprocessing work done on the data included:
1) Outlier removal 
2) Label and one hot encoding
3) Handling missing data by mode imputation
4) Handling imbalanced data by oversampling using SMOTE, 
5) Dimensionality reduction
6) Normalization and standardization 

# Models
Classsifiers used:
1) Support Vector Machine (SVM)
2) Naive Bayes
3) K Nearest Neighbors
4) Random Forest
5) Perceptron

# Results
Performance Evaluation Metric used:
1) F1 score
2) AUC score
3) Training and test accuracy
4) Confusion matrix
5) ROC plots

