# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from src.normal_equation import compute_theta
from src.gradient_descent import descent
from src.feature_normalize import normalize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

# %% [markdown]
# # House Pricing

# This was an assignment on Andrew Ng's Machine Learning Course. I've already done the exercise in Octave, this project is to help me get familiar with the different libraries in Python.

# ## Problem

# Suppose you are selling your house and you want to know what a good market price would be. One way to do this is to first collect information on recent houses sold and make a model of housing prices.

# ## Data
# The file **data.txt** contains a training set of housing prices in Port- land, Oregon. The first column is the size of the house (in square feet), the second column is the number of bedrooms, and the third column is the price of the house.

# %% Reading data
df = pd.read_csv('data/data.txt', names=['Size', 'Bedrooms', 'Price'])
df.head()

# %% [markdown]
# ## Feature Normalization
# Looking at the data, it is clear that there's a huge difference in the two features. This will have a significant effect in running Gradient Descent, remember that the formula requires minimizing the product of features and a value theta. The algorithm might think that the feature *Size* have a significantly higher effect in Price compared to the *Number of Bedrooms*
#
# This can be addressed by scaling the features using the *mean* and *standard deviations*.
#
# First create an X matrix containing all the samples in rows and its features in columns. Put the actual price values in a separate Y vector.

# %%
# Create the X-matrix
X = df.iloc[:, :len(df.columns)-1].to_numpy()
# Create the Y-matrix
Y = df.iloc[:, -1].to_numpy()

# %% [markdown]
# Using the *feature_normalize* script, pass in the X matrix. The script will handle all feature scaling using each features *mean* and *standard deviation*.
#
# Store the result in a separate matrix, in case the original X values are needed.
#
# ```python
# def normalize(X: np.array):
#     mu = np.mean(X, axis=0)
#     std = np.std(X, axis=0, ddof=1)
#     return [(X - mu) / std, mu, std]
# ```

# %%
X_norm, mu, std = normalize(X)

# %% [markdown]
# ## Gradient Descent
# After the features have been normalized, prepare the variables required to perform Gradient Descent.
#
# First, add in the 1's column for the X-intercept. Initalized the theta values to 0. Make sure to experiment with the learning rate to see the best result.
#

# %% Variables for Gradient Descent
m, n = X.shape
X_1 = np.c_[np.ones(m), X_norm]
alpha = 0.03
iterations = 400
theta = np.zeros(n + 1)

# %% [markdown]
# Run Linear Regression with Gradient Descent using the *gradient_descent* script.
# This can now then minimize the JCost function using the theta values.
#
# ```python
# def descent(X: np.array, Y: np.array, theta: np.array, alpha):
#     m = len(Y)
#     h = X.dot(theta)
#     theta = theta - ((h - Y).dot(X) * (alpha / m))
#     return theta
# ```

# %% Gradient Descent
for i in range(iterations):
    theta = descent(X_1, Y, theta, alpha)
print(f'Computed theta (Gradient Descent): {theta}')

# %% [markdown]
# ## Normal Equation
# The computed thetas can now then be used to predict house prices given it's size and number of bedrooms.
#
# In this section, Normal Equation will be demonstrated. It is a form of Linear Regression that doesn't use batch updates like Gradient Descent.
# There are several pros and cons to the two algorithms, ultimately it will vary from use case to use case.
#
# Here is the closed form equation of Linear Regression:
# ![normal_equation](./docs/normal_equation.png)
#
# Using the *normal_equation* script, import the *compute_theta* function. This function will only require the feature matrix X and Y.
#
# ```python
# def compute_theta(X: np.array, Y: np.array):
#     xtx_inv = np.linalg.pinv(X.transpose().dot(X))
#     xty = X.transpose().dot(Y)
#     return xtx_inv.dot(xty)
# ```

# %%
X_1_norm = np.c_[np.ones(m), X]
theta_normal = compute_theta(X_1_norm, Y)
print(f'Computed theta (Normal Equation): {theta_normal}')

# %% [markdown]
# ## Prediction
# Using the two theta values computed, we can now predict the price for a house given it's size and number of bedrooms.
#
# When using Gradient Descent, it's important to remember to normalize the feature parameters first before making a prediction.

# %%
house_size = 1650
house_rooms = 3

features = np.array([house_size, house_rooms])
scaled = (features - mu) / std
x_gradient = np.append([1], scaled)
x_normal = np.array([1, house_size, house_rooms])

price_gradient = x_gradient.dot(theta)
price_normal = x_normal.dot(theta_normal)

print(f'Price of House with {house_size} sqr feet and {house_rooms} br\n')
print(f'Gradient Descent: ${price_gradient:,.2f}')
print(f'Normal Equation: ${price_normal:,.2f}')
