❤️ Heart Disease Prediction using Machine Learning

📌 Problem Statement

The objective of this project is to build and compare multiple Machine Learning classification models to predict whether a person has heart disease based on medical attributes. The goal is to evaluate different models and identify the best-performing one using standard performance metrics.

📊 Dataset Description

The dataset used in this project was obtained from Kaggle – “Heart Failure Prediction Dataset” by Fedesoriano.
It contains real clinical records collected from multiple hospitals and focuses on predicting the occurrence of heart disease events.

The dataset consists of 918 patient records with 11 clinical input features and 1 binary target variable (HeartDisease).
These features represent important physiological and medical indicators commonly used by cardiologists for diagnosis.

🔹 Features include:

Age

Sex

ChestPainType

RestingBP

Cholesterol

FastingBS

RestingECG

MaxHR

ExerciseAngina

Oldpeak

ST_Slope

🔹 Target Variable:

HeartDisease

0 → No heart disease

1 → Heart disease present

🔹 Why this dataset is important:

Based on real clinical heart failure data

Contains medically meaningful features

Balanced enough for ML evaluation

Widely used for benchmarking ML classifiers


⚙️ Machine Learning Models Implemented

Logistic Regression

Decision Tree

K-Nearest Neighbors (KNN)

Naive Bayes

Random Forest

XGBoost
__________________________________________________________________________________________________
| ML Model Name           | Accuracy  | AUC       | Precision | Recall    | F1        | MCC       |
| ------------------------| --------- | --------- | --------- | --------- | --------- | --------- |
| Logistic Regression     | 0.853     | 0.927     | 0.900     | 0.841     | 0.870     | 0.704     |
| Decision Tree           | 0.826     | 0.823     | 0.857     | 0.841     | 0.849     | 0.644     |
| KNN                     | 0.853     | 0.924     | 0.900     | 0.841     | 0.870     | 0.704     |
| Naive Bayes             | 0.859     | 0.930     | 0.909     | 0.841     | 0.874     | 0.717     |
| Random Forest(Ensemble) | 0.875     | 0.935     | 0.896     | 0.888     | 0.892     | 0.744     |
| XGBoost (Ensemble)      | 0.870     | 0.921     | 0.895     | 0.879     | 0.887     | 0.733     |
|_________________________|___________|___________|___________|___________|___________|___________|


__________________________________________________________________________________________________
| Model Name          | Observation about model performance                                      |
| ------------------- | ------------------------------------------------------------------------ |
| Logistic Regression | Provided a strong and stable baseline performance across all metrics.    |
| Decision Tree       | Showed lower accuracy and generalization compared to ensemble models.    |
| KNN                 | Achieved good performance but was sensitive to feature scaling.          |
| Naive Bayes         | Delivered fast training with competitive accuracy and AUC values.        |
| Random Forest       | **Achieved the best overall performance across all evaluation metrics.** |
| XGBoost             | Performed very well with high accuracy and AUC, close to Random Forest.  |
|_____________________|__________________________________________________________________________|
