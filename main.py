import pandas as ps
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset
default = pd.read_csv('default.csv')

# Inspect the dataset
default.head()

# Convert the student attribute into dummy variables 0s and 1s
default = pd.get_dummies(default, drop_first=True)

# Set the RNG seed so that we get reproducible results
# Splitting is random process
# Seed value doesn't really matter
np.random.seed(0)

# Grab our input and output data
x_data = default[default.columns.difference(['default'])]
y_data = default['default']

# We can now split this into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3)

# Initialize the model without changing any of the default parameters
model = LogisticRegression()

# Fit the model on our training data
model.fit(x_train, y_test)

# Find how well we do on a testing set
accuracy_score = model.score(x_test, y_test)
accuracy_score

# Predict on our testing set
y_pred = model_2.predict(x_test)

# Compare my prediction with the actual data
print(accuracy_score(y_pred, y_test))

# TODO: Print the confusion matrix
print(.....)
