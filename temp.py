# -*- coding: utf-8 -*-
"""
Spyder Editor
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

dataset = pd.read_csv("Salary_Data.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


imp = SimpleImputer(missing_values=np.nan, strategy="mean")
x = imp.fit_transform(x)
y = y.reshape(-1, 1)
y = imp.fit_transform(y)
y = y.reshape(-1)

# Data pre-processiing for categorical data
cat_dataset = pd.read_csv("Data.csv")
xx = pd.DataFrame(cat_dataset.iloc[:, :-1].values)
yy = pd.DataFrame(cat_dataset.iloc[:, -1].values)
# Dealing with missing values
imp = imp.fit(xx.values[:, 1:3])
xx.values[:, 1:3] = imp.transform(xx.values[:, 1:3])

# Dealing with categorical data

labelencoder_xx = LabelEncoder()
xx.values[:, 0] = labelencoder_xx.fit_transform(xx.values[:, 0])
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(), [0])],    # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
    remainder='passthrough'                         # Leave the rest of the columns untouched
)
xx = np.array(ct.fit_transform(xx), dtype=np.float)
labelencoder_yy = LabelEncoder()
yy = labelencoder_yy.fit_transform(yy)

# Continuing with first dataset i.e. salary_dataset for Training and Testing
# Splitting the dataset into the Training set and Test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# Regression
reg = LinearRegression()
reg.fit(x_train, y_train)

# for predict the last values
y_predict = reg.predict(x_test)

# Visualize the Training Data
plt.scatter(x_train, y_train, color="red")
plt.plot(x_train, reg.predict(x_train), color="blue")
plt.title("Linear Regression Salary vs Experience")
plt.xlabel("Experience in Years")
plt.ylabel("Salary")
plt.show()

# Visualize the Testing Data
plt.scatter(x_test, y_test, color="red")
plt.plot(x_train, reg.predict(x_train), color="blue")
plt.title("Testing Linear Regression Salary vs Experience")
plt.xlabel("Experience in Years")
plt.ylabel("Salary")
plt.show()