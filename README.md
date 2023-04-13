# Heart_Disease_Prediction
## Overview:

This project aims to explore and analyze the Heart Disease prediction dataset, obtained from Kaggle, using various machine learning methods for classification. The dataset includes features related to different medical attributes, and the task is to predict whether a person has heart disease or not, which is a binary classification problem.

The project follows the following steps:

* Pre-processing, Data Mining, and Visualization: The dataset is visualized and explored using various plots, linear regression, PCA, and clustering techniques to identify any natural relationships or correlations among variables. The features that are chosen as input variables are determined based on their relevance and importance in predicting heart disease. Pre-processing techniques such as handling missing values, feature scaling, and encoding categorical variables are applied to prepare the data for machine learning algorithms. Strongly correlated variables, both positively and negatively, are identified to avoid multicollinearity issues.

* Classification: Different machine learning methods are implemented to solve the classification task. These methods include algorithms such as logistic regression, decision trees, random forests, support vector machines, and neural networks. The parameters for each method are tuned and optimized to achieve the best performance. Additional features such as polynomial features or derived features are also considered to improve the model's predictive power.

* Evaluation: The performance of each classification method is evaluated using various metrics such as confusion matrix, F1 value, bias, and variance. Receiver Operator Curve (ROC) is generated for at least one of the methods to analyze the classifier's performance at different operating points. The best performing classifier is determined based on the evaluation statistics and the chosen performance metric.

* Iteration: One of the classifiers is selected, and at least three iterations are performed to make modifications and improve its performance on a chosen statistic. The impact of each modification on the performance metric is analyzed using the confusion matrix and other evaluation metrics.

The results of the analysis, including the evaluation statistics, ROC curves, and modifications made during iterations, are included in the report to provide a comprehensive understanding of the project's findings and conclusions.
