**Heart Disease Prediction**

This repository contains a heart disease prediction machine learning model. It leverages four different algorithms for model training: XGBoost, Support Vector Machine (SVM), Random Forest, and Logistic Regression.

**Table of Contents**

Introduction

Dataset

Dependencies

Installation

Usage

Model Training

Evaluation


**Introduction**

Heart disease is one of the leading causes of death globally. Early detection through predictive modeling can help in taking preventive measures and saving lives. This project aims to predict the presence of heart disease in patients using various machine learning models.

**Dataset**

The dataset used for this project is sourced from Kaggle, specifically the "Heart Disease" dataset. The dataset consists of several features that include patient information such as age, sex, chest pain type, resting blood pressure, cholesterol levels, fasting blood sugar, resting electrocardiographic results, maximum heart rate achieved, exercise-induced angina, ST depression induced by exercise, the slope of the peak exercise ST segment, number of major vessels, and a few others.

**Dependencies**

Python 3.x

NumPy

Pandas

Scikit-learn

XGBoost

Matplotlib

You can install the required dependencies using pip:
pip install numpy pandas scikit-learn xgboost matplotlib seaborn

**Installation**

Clone the repository:
bash
Copy code
git clone https://github.com/its-akanksha/Heart-Disease-Prediction.git
Navigate to the project directory:

bash
Copy code
cd heart-disease-prediction

Ensure you have installed all dependencies as specified.

Prepare your dataset (make sure it's in the correct format as expected by the models).

Run the script to train the models and evaluate their performance.

bash
Copy code
python heart_disease.py

**Model Training**

The models used for training in this project are:

XGBoost: An efficient and scalable implementation of gradient boosting framework by Friedman et al. (2000).

Support Vector Machine (SVM): A supervised machine learning algorithm which can be used for classification or regression challenges.

Random Forest: A versatile machine learning method capable of performing both regression and classification tasks.

Logistic Regression: A statistical model that in its basic form uses a logistic function to model a binary dependent variable.

Each model is trained on the dataset, and their performance is evaluated using appropriate metrics.

**Evaluation**
The models are evaluated based on several metrics:

Accuracy

Precision

Recall

F1 Score

Classification_report

The evaluation results are printed in the console and saved to a file for further analysis.
