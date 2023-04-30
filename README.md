# Telco-Customer-Churn-Predictions

Introduction: In this data science project, we aimed to develop a model that can predict customer churn for a telecommunications company. Churn prediction is a critical problem for many companies as losing customers can significantly impact revenue and profitability. We used various machine learning algorithms, including logistic regression, random forest, gradient boosting, XGBoost, LightGBM, K-Nearest Neighbors, and soft voting classifiers, to predict churn. We evaluated the performance of these models using different performance metrics such as accuracy, precision, recall, and AUC. In this report, we present our findings, including the best performing model and its corresponding evaluation metrics.

Data Preprocessing: We began the project by performing exploratory data analysis to understand the nature of the data and its distributions. We then conducted data cleaning, which involved handling missing values, removing duplicates, and converting categorical variables to numeric using one-hot encoding. Additionally, we balanced the data by undersampling the majority class.
Model Selection and Evaluation: We used seven different machine learning algorithms to predict churn, including logistic regression, random forest, gradient boosting, XGBoost, LightGBM, K-Nearest Neighbors, and soft voting classifiers. We evaluated these models using different performance metrics such as accuracy, precision, recall, and AUC.

Performance Comparison: The table below shows the performance comparison of different classification models for churn prediction.
Model	Accuracy	Precision	Recall	AUC
<br>LR	0.789	0.566	0.699	0.838
RF	0.754	0.509	0.741	0.838
GBC	0.724	0.471	0.786	0.827
XGB	0.723	0.471	0.792	0.829
LGBM	0.760	0.518	0.727	0.837
KNN	0.751	0.504	0.656	0.768
SVOT	0.847	0.799	0.925	0.925

The soft voting classifier (SVOT) performed the best with the highest accuracy, precision, recall, and AUC. It outperformed all other models by a significant margin. The random forest (RF) model performed the second best, followed by logistic regression (LR) and LightGBM (LGBM).

ROC Curve: The ROC curve provides a visual representation of the comparative performance of the different models based on the selected metrics. The plot below shows the ROC curve for the different models.

From the ROC curve, we can see that the soft voting classifier (SVOT) outperforms all other models in terms of the AUC score.

Cumulative Gains Curve: The cumulative gains curve is a useful tool in evaluating the performance of a binary classifier, especially in cases where there is a class imbalance. The plot below shows the cumulative gains curve for the soft voting classifier.

From the cumulative gains curve, we can see that with only 20% of the population targeted, the classifier is able to capture approximately 80% of the positive samples.

Conclusion: In conclusion, we developed a model to predict customer churn for a telecommunications company. We used various machine learning algorithms and evaluated their performance using different performance metrics such as accuracy, precision, recall, and AUC. The soft voting classifier (SVOT) outperformed all other models with the highest accuracy, precision, recall, and AUC. The random forest (RF) model performed the second best, followed by logistic regression (

