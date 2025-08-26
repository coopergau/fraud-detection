# Credit Card Fraud Detection

## Overview

The purpose of this project is to explore and compare the performance of different machine learning techniques for detecting fraudulent credit card transactions.


## Dev
Key Techniques to Showcase:

Handle class imbalance (SMOTE, undersampling, cost-sensitive learning)
Multiple evaluation metrics (precision, recall, F1, AUC-ROC, AUC-PR)


1. Load, inspect, clean data 
2. Exploratory Data Analysis 
3. Feature Engineering 
4. Handle Class Imbalance 
5. Split into train/val/test 
6. Train baseline model 
7. Hyperparameter tuning with validation 
8. Evaluate with proper metrics 
9. Feature importance analysis 
10. Final test set evaluation 

### To do at end
- make comments nice in notebook (I think you can use md formatting or something)
- mention the choice of scalers (robust is better for stuff with big outliers i think)
- check that the xgboost has eval_metric="logloss"
- choose your own Hyperparameters
- what is pos_weight_ratio
- review the roc and pr curves and how to explain those