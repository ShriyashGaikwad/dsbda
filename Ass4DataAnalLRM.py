import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("boston.csv")
df.head()

df.info()

df.info()

X = df.drop("MEDV", axis=1)
y = df["MEDV"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

y_pred[:5]

X_rm = df[['RM']]
y_medv = df['MEDV']

simple_model = LinearRegression()
simple_model.fit(X_rm, y_medv)

line = simple_model.predict(X_rm)

errors = y_test - y_pred

mean_squared_error(y_test, y_pred)

r2_score(y_test, y_pred)