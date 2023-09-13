from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier



def Clean(X_data,Y_data):

    n = 5
    Error =  2

    knn = KNeighborsClassifier(n_neighbors = n, weights = 'distance')
    knn.fit(X_data,Y_data)
    
    counter = 0


    
    for i in X_data:
        #print("Probability: ",knn.predict_proba([i])[0][0])
        if(knn.predict_proba([i])[0][0]>= ((n-Error)/n) and Y_data[counter] == 1):
         X_data =np.delete(X_data,counter,0)
         Y_data =np.delete(Y_data,counter,0)
         counter -=1
         #print("Probability: ",knn.predict_proba([i])[0][0])

        if(knn.predict_proba([i])[0][0]<=(Error/n) and Y_data[counter]==0):
         X_data =np.delete(X_data,counter,0)
         Y_data =np.delete(Y_data,counter,0)
         #print("Probability: ",knn.predict_proba([i])[0][0])
         counter -=1
        counter += 1
      
    #NewX.reshape(-1, 1)
    
    return X_data , Y_data

#place file name here
#1 is Male, 0 is Female
dataset = pd.read_csv("test.csv")

# change numbers n stuff
X = dataset.iloc[:, [2,3]].values
  
# change number n stuff
y = dataset.iloc[:, 4].values


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = .25, random_state=0)
sc_x = StandardScaler()


for i in range(0):
    X_train, y_train = Clean(X_train,y_train)

#print(len(X_train))

X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)
poly = PolynomialFeatures(degree = 3 , interaction_only=False, include_bias=False)
X_poly = poly.fit_transform(X_train)
lr = LogisticRegression()
lr.fit(X_poly,y_train)
X_test_Poly = poly.transform(X_test)




print(((lr.score(X_test_Poly, y_test))*100),"%")

X_set, y_set = X_train, y_train

X1, X2 = np.meshgrid(np.arange(start = X_train[:, 0].min() - 1, stop = X_train[:, 0].max() + 1, step = 0.01),np.arange(start = X_train[:, 1].min() - 1, stop = X_train[:, 1].max() + 1, step = 0.01))

X_grid = poly.transform(np.column_stack((X1.ravel(), X2.ravel())))

y_pred = lr.predict(X_grid)
y_pred = y_pred.reshape(X1.shape)

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())



#plt.xlim(dataset.iloc[:, 2].values.min(), dataset.iloc[:, 2].values.max()) # Creates New range for data
#plt.ylim(dataset.iloc[:, 3].values.min(), dataset.iloc[:, 3].values.max())# Creates New range for data

plt.contourf(X1, X2, y_pred, alpha = 0.75, cmap = ListedColormap(('red', 'green')))
  
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],c = ListedColormap(('red', 'green'))(i), label = j)
      
plt.title('Classifier (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()




