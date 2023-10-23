import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# Loading Diabetes dataset
d = datasets.load_diabetes()

# Names of the features
d.feature_names


# X and y data
X = d.data
y = d.target

# Printing a couple of samples for X and y
print(X[:3])
print()
print(y[:3])


# Dividing data into train and test sets with a test portion of 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Checking the dimensions of the train and test sets
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# Creating the Linear model object
model = linear_model.LinearRegression()


# Training the model
model.fit(X_train, y_train)


# Prediction on the test data
y_pred = model.predict(X_test)


# Plotting the actual and predicted values
plt.scatter(y_test, y_pred, color='red')
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
# If actual values and prdicted values form a shape as much as close to a line, it means the predicted values are
# close to the actual values
plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()], color='blue', linewidth=2);


# Calculating Mean Squared Error (MSE) to check the model's performance
mse = mean_squared_error(y_test, y_pred)
print(f"The Mean Squared Error is: {mse:.2f}")