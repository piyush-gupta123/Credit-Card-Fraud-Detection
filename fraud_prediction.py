# Importing all the dependencies required
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# Creating dataframe using pandas library
data = pd.read_csv('creditcard.csv')
print(data.shape)
print(data.tail())

# Data Preprocessing
print(data.isnull().any())

print(data['Class'].value_counts())

valid = data[data.Class == 0]
fraud = data[data.Class == 1]
print(len(valid))
print(len(fraud))

print(valid.Amount.describe())
print(fraud.Amount.describe())

print(data.groupby('Class').mean())

# Undersampling to get equal amount of data to train to avoid any problems
valid_sample = valid.sample(n=492)
new_data = pd.concat([valid_sample, fraud], axis=0)
print(new_data['Class'].value_counts())
print(new_data.groupby('Class').mean())

#Separating Data into features and labels
X = new_data.drop(columns='Class', axis=1)
Y = new_data['Class']


# Splitting data into train and test for training and testing
x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=2)
print(x_train.shape)
print(x_test.shape)


# First model - Logistic Regression

model = LogisticRegression()
model.fit(x_train, y_train)

# Training Accuracy of model
train_prediction = model.predict(x_train)
train_accuracy = accuracy_score(train_prediction, y_train)
print('Training Data Accuracy - Logistic Regression :', train_accuracy)


# Testing Accuracy of model
test_prediction = model.predict(x_test)
test_accuracy = accuracy_score(test_prediction, y_test)
print('Testing Data Accuracy - Logistic Regression :', test_accuracy)

# Confusion Matrix for Logistic Regression
c_matrix = confusion_matrix(y_test,test_prediction)
print('Confusion Matrix for Logistic Regression:\n ',c_matrix)



# Second Model - Random Forest Classifier

model = RandomForestClassifier()
model.fit(x_train, y_train)


# Training Accuracy for model
train_prediction = model.predict(x_train)
train_accuracy = accuracy_score(train_prediction, y_train)
print('Training Data Accuracy - Random Forest :', train_accuracy)

# Testing Accuracy for model
test_prediction = model.predict(x_test)
test_accuracy = accuracy_score(test_prediction, y_test)
print('Testing Data Accuracy - Random Forest :', test_accuracy)

# Confusion Matrix For Random Forest classifier
c_matrix = confusion_matrix(y_test,test_prediction)
print('Confusion Matrix for Random Forest:\n ',c_matrix)
