# Telco Customer Churn Predictions

![Python](https://img.shields.io/badge/Python-306998?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-FF6F00?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-3C7EBB?style=for-the-badge&logo=pandas&logoColor=white)
![Numpy](https://img.shields.io/badge/Numpy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-009688?style=for-the-badge&logo=matplotlib&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-FF6F00?style=for-the-badge&logo=xgboost&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-FB8C00?style=for-the-badge&logo=lightgbm&logoColor=white)



## Introduction

In this data science project, we aimed to develop a model that can predict customer churn for a telecommunications company. Churn prediction is critical for many companies as losing customers can significantly impact revenue and profitability. We used various machine learning algorithms, including logistic regression, random forest, gradient boosting, XGBoost, LightGBM, K-Nearest Neighbors, and soft voting classifiers, to predict churn. We evaluated the performance of these models using different performance metrics such as accuracy, precision, recall, and AUC. This report presents our findings, including the best performing model and its corresponding evaluation metrics.

## Data Preprocessing

We began the project by performing exploratory data analysis to understand the nature of the data and its distributions. The data preprocessing steps included:
- Handling missing values
- Removing duplicates
- Converting categorical variables to numeric using one-hot encoding
- Balancing the data by undersampling the majority class

## Model Selection and Evaluation

We used seven different machine learning algorithms to predict churn:
1. Logistic Regression (LR)
2. Random Forest (RF)
3. Gradient Boosting Classifier (GBC)
4. XGBoost (XGB)
5. LightGBM (LGBM)
6. K-Nearest Neighbors (KNN)
7. Soft Voting Classifier (SVOT)

We evaluated these models using various performance metrics: accuracy, precision, recall, and AUC.

## Performance Comparison

The table below shows the performance comparison of different classification models for churn prediction:

| Model | Accuracy | Precision | Recall | AUC |
|-------|----------|-----------|--------|-----|
| LR    | 0.789    | 0.566     | 0.699  | 0.838 |
| RF    | 0.754    | 0.509     | 0.741  | 0.838 |
| GBC   | 0.724    | 0.471     | 0.786  | 0.827 |
| XGB   | 0.723    | 0.471     | 0.792  | 0.829 |
| LGBM  | 0.760    | 0.518     | 0.727  | 0.837 |
| KNN   | 0.751    | 0.504     | 0.656  | 0.768 |
| SVOT  | 0.847    | 0.799     | 0.925  | 0.925 |

The soft voting classifier (SVOT) performed the best with the highest accuracy, precision, recall, and AUC, significantly outperforming all other models. The random forest (RF) model performed the second best, followed by logistic regression (LR) and LightGBM (LGBM).

## ROC Curve

The ROC curve provides a visual representation of the comparative performance of the different models based on the selected metrics. From the ROC curve, we can see that the soft voting classifier (SVOT) outperforms all other models in terms of the AUC score.


## Cumulative Gains Curve

The cumulative gains curve is a useful tool in evaluating the performance of a binary classifier, especially in cases where there is a class imbalance. The plot below shows the cumulative gains curve for the soft voting classifier. From the cumulative gains curve, we can see that with only 20% of the population targeted, the classifier is able to capture approximately 80% of the positive samples.


## Conclusion

In conclusion, we developed a model to predict customer churn for a telecommunications company. We used various machine learning algorithms and evaluated their performance using different performance metrics such as accuracy, precision, recall, and AUC. The soft voting classifier (SVOT) outperformed all other models with the highest accuracy, precision, recall, and AUC. The random forest (RF) model performed the second best, followed by logistic regression (LR) and LightGBM (LGBM).

## Acknowledgements
- Scikit-Learn
- Pandas
- Numpy
- Matplotlib
- XGBoost
- LightGBM
