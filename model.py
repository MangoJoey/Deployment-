import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

height_weight = pd.read_csv("hw_200.csv")

X = height_weight[[' Height(Inches)"']].values
y = height_weight[[' "Weight(Pounds)"']].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
regressor = LinearRegression()
regressor.fit(X_train, y_train)

pickle.dump(regressor, open('model.pkl', 'wb'))
