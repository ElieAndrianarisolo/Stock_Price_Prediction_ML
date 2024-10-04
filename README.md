# README: Stock Price Prediction using Machine Learning

## Project Overview

This project implements a machine learning solution to predict stock prices using historical data. It tackles the problem from both a classification and regression perspective:

- **Classification Task**: Predict whether the stock price will increase (Buy signal) or decrease (Sell signal) the next day.
- **Regression Task**: Predict the closing price of the stock for the next day.

The model utilizes the K-Nearest Neighbors (KNN) algorithm for both classification and regression tasks. GridSearchCV is applied to fine-tune the hyperparameters of the KNN model and improve accuracy.

## Dataset

The dataset used for this project is historical stock data for the stock **TATAGLOBAL**, stored in a CSV file (`NSE-TATAGLOBAL.csv`). The dataset contains the following columns:

- **Date**: Date of the stock data entry.
- **Open**: Stock's opening price for the day.
- **Close**: Stock's closing price for the day.
- **High**: Highest price the stock traded for during the day.
- **Low**: Lowest price the stock traded for during the day.

## Problem Definition

### 1. Classification Task

The objective of this task is to determine whether the stock should be **bought (+1)** or **sold (-1)** based on the historical price data.

- **Input Features**:
  - `Open - Close`: The difference between the opening and closing prices.
  - `High - Low`: The difference between the highest and lowest prices of the day.

- **Target Variable**: 
  - A binary signal indicating whether the stock price will rise or fall on the next day. If the next day's closing price is higher than the current day's closing price, it will output **+1** (Buy), otherwise **-1** (Sell).

### 2. Regression Task

The objective of this task is to predict the exact **closing price** of the stock for the next day.

- **Input Features**: 
  - The same features as in the classification task (`Open - Close`, `High - Low`) are used.

- **Target Variable**: 
  - The closing price of the stock.

## Methodology

### Classification Task (Buy/Sell Prediction)

1. **Data Preprocessing**:
   - The CSV data is loaded into a pandas DataFrame, and the Date column is set as the index.
   - Two new features (`Open - Close` and `High - Low`) are computed.
   - Missing values are removed.

2. **Model Selection**:
   - K-Nearest Neighbors (KNN) algorithm is used for classification.
   - GridSearchCV is employed to tune the `n_neighbors` hyperparameter, optimizing the number of neighbors used for predictions.

3. **Model Training & Evaluation**:
   - The dataset is split into training (75%) and testing (25%) sets.
   - The KNN model is trained on the training set, and the accuracy is evaluated on both the training and testing sets.
   - The results are printed in terms of accuracy.

### Regression Task (Stock Price Prediction)

1. **Data Preprocessing**:
   - The same features and target variable (closing price) are used.
   
2. **Model Selection**:
   - K-Nearest Neighbors (KNN) algorithm is used for regression.
   - Similar to the classification task, GridSearchCV is applied to optimize the `n_neighbors` hyperparameter.

3. **Model Training & Evaluation**:
   - The data is split into training (75%) and testing (25%) sets.
   - The model is trained on the training set, and predictions are made on the test set.
   - The **Root Mean Squared Error (RMSE)** is used to evaluate the accuracy of the predicted closing prices.

## Conclusion

This project demonstrates the application of K-Nearest Neighbors for stock price prediction. The classification model helps determine whether to buy or sell a stock, while the regression model predicts the exact closing price. Fine-tuning the model with GridSearchCV helps improve prediction accuracy for both tasks.
