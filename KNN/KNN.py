import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sb
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

#place file name here
#1 is Male, 0 is Female

num_of_Neighbors = 1

dataset = pd.read_csv("test.csv", names=['ID', 'Gender', 'Age', 'Salary', 'Bought'])

X = dataset[['Age', 'Salary']].values
Y = dataset['Bought'].values
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.25,random_state=0)
knn = KNeighborsClassifier(n_neighbors=num_of_Neighbors)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
num_wrong = np.sum(np.not_equal(knn.predict(X_test), y_test))
print(f'We got {num_wrong} wrong!')
print(knn.score(X_test, y_test))

