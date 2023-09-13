#Import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
import random

#sigmoid function
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def accuracy(y_pred, y_test):
  return np.sum(y_pred==y_test)/len(y_test)


class LogisticRegression():

  #initialization
  def __init__(self, lr=0.01, numIter=1000):
    #set the learn rate
    self.lr = lr
    #set the number of iterations
    self.numIter = numIter
    #default weights
    self.weights = None
    #default bais
    self.bias = None

  def fit(self, x, y):
    numSamples, numFeatures = x.shape
    #assign weights and bais as 0
    self.weights = np.zeros(numFeatures)
    self.bias = 0
    #predicting the result using sigmoid function
    for i in range(self.numIter):
      #the linear predictions
      #dot product
      lines = np.dot(x, self.weights) + self.bias
      #inputting the linear prodictions in sigmoid to get the actual predictions
      prediction = sigmoid(lines)

      #calculating gradience for weight and bais
      #gradient of weight         #dot product, x.T = transpose of x
      gradientW = (1 / numSamples) * np.dot(x.T, (prediction - y))
      #gradient of bais
      gradientB = (1 / numSamples) * np.sum(prediction - y)

      #Now update weight and bais
      self.weights = self.weights - self.lr * gradientW
      self.bias = self.bias - self.lr * gradientB

  def predict(self, x):

    lines = np.dot(x, self.weights) + self.bias
    #the y predictions
    y_pred = sigmoid(lines)
    #returns 0 if less than 0.5 or 1 if more than 0.5
    return [1 if y <= 0.5 else 0 for y in y_pred]
def main():
 
  '''bc = datasets.load_breast_cancer()
  X, y = bc.data, bc.target
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 1234)
  model = LogisticRegression()
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)
  acc = accuracy(y_pred, y_test)
  print(acc)'''
  
  
  bc = pd.read_csv("test.csv")
  X = bc.iloc[:, 2].values
  y = bc.iloc[:, 4].values
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 1234)
  model = LogisticRegression()
  model.fit(X_train, y_train,1)
  y_pred = model.predict(X_test)
  acc = accuracy(y_pred, y_test)
  print(acc)
main()