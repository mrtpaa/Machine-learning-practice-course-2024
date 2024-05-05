#------ML course final assignment------
#The final module of this course is a project to determine which basketball teams are most likely 
#to make it to the semifinal round of the College Basketball Tournament known as the Final Four.

import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
%matplotlib inline

#Read in the file
df = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%206/cbb.csv')
df.head()

#Check the shape
df.shape

# Wins Above Bubble (refers to the cut off between making the NCAA March Madness Tournament & not making it).
# Add a column that will contain "true" if the wins above bubble are over 7 and "false" if not. 
# We'll call this column Win Index or "windex" for short.
df['winindex']= np.where(df.WAB >7, 'True', 'False')
df.head()

#Filter the data set to the teams that made the Sweet Sixteen, the Elite Eight, and the Final Four 
#in the post season. We'll also create a new dataframe that will hold the values with the new column.
df1 = df.loc[df['POSTSEASON'].str.contains('S16|E8|F4', na = False)]
df1.head()

#how many teams are there
df1['POSTSEASON'].value_counts()


###Plot some data to understand it.
#Plot chance of winning an average Division I team

import seaborn as sns
bins = np.linspace(df1.BARTHAG.min(), df1.BARTHAG.max(), 10) #
g = sns.FacetGrid(df1, col="winindex", hue="POSTSEASON", palette="Set3", col_wrap=6)
g.map(plt.hist, 'BARTHAG', bins=bins, ec="k")
g.axes[-1].legend()
plt.show()

#Adjusted Offensive Efficiency (An estimate of the offensive efficiency 
#(points scored per 100 possessions) a team would have against the average Division I defense

bins = np.linspace(df1.ADJOE.min(), df1.ADJOE.max(), 10)
g = sns.FacetGrid(df1, col="winindex", hue="POSTSEASON", palette="Set2", col_wrap=2)
g.map(plt.hist, 'ADJOE', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


#Adjusted Defensive Efficiency (An estimate of the offensive efficiency 
#(points scored per 100 possessions) a team would have against the average Division I defense
bins = np.linspace(df1.ADJDE.min(), df1.ADJDE.max(), 10)
g = sns.FacetGrid(df1, col="winindex", hue="POSTSEASON", palette="husl", col_wrap=2)
g.map(plt.hist, 'ADJDE', bins=bins, ec="k")
g.axes[-1].legend()
plt.show()

#this doesnt impact much to get the team in the final four


####-----Preprocessing------

#Let's look at the postseason#
df1.head()
df1.groupby(['winindex'])['POSTSEASON'].value_counts(normalize=True)
df1.head()


# 13% of teams with 6 or less wins above bubble make it into the final four while 17% of teams with 7 or more do.
#Lets convert wins above bubble (winindex) under 7 to 0 and over 7 to 1:
df1['winindex'].replace(to_replace=['False','True'], value=[0,1],inplace=True)
df1.head()

# These inputs are the parameters to train the  data set.
X = df1[['G', 'W', 'ADJOE', 'ADJDE', 'BARTHAG', 'EFG_O', 'EFG_D',
       'TOR', 'TORD', 'ORB', 'DRB', 'FTR', 'FTRD', '2P_O', '2P_D', '3P_O',
       '3P_D', 'ADJ_T', 'WAB', 'SEED', 'winindex']]
X[0:5]


#This is what we want to predict
y = df1['POSTSEASON'].values
y[0:10]

## normalize the data
X= preprocessing.StandardScaler().fit(X).transform(X)
X

#Train the data set
from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(X, y, test_size = 0.2, random_state =4)
print('Train set:', X_train.shape, Y_train.shape)
print('Test set:', X_val.shape, Y_val.shape)


#-------Here starts the classification---------
#Now, it is your turn, use the training set to build an accurate model. Then use the validation set to report the accuracy of the model You should use the following algorithm:

# 1) K Nearest Neighbor(KNN)
# 2) Decision Tree
# 3) Support Vector Machine
# 4) Logistic Regression

#-------- Build a KNN model using a value of k = 5, 
#--------find the accuracy on the validation data (X_val and y_val)!
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
#Train the model and predict
k=5
neighbors = KNeighborsClassifier(n_neighbors = k).fit(X_train, Y_train) 
neighbors

yhat = neighbors.predict(X_val)
yhat
xhat = neighbors.predict(X_train)
xhat

#calculate the accuracy of the prediction
from sklearn import metrics
print("Train set accuracy:", metrics.accuracy_score(Y_train, xhat))
print("Test set accuracy: ", metrics.accuracy_score(Y_val, yhat))

print(Y_val)


#-------Build a DECISION TREE algorigthm
from sklearn.tree import DecisionTreeClassifier
winningtree = DecisionTreeClassifier(criterion= 'entropy', max_depth = 4)
winningtree

winningtree.fit(X_train, Y_train)
yhat_predictionTree = winningtree.predict(X_val)
print(predictionTree)
print(Y_val)
print("Decisiontree accuracy: ", metrics.accuracy_score(Y_val, yhat_predictionTree))

!conda install -c conda-forge pydotplus -y
!conda install -c conda-forge python-graphviz -y

import matplotlib.pyplot as plt
from sklearn import tree
tree.plot_tree(winningtree)
plt.show()

#--- Make a for loop to check for maximal depth 
Ks = 10
mean_accuracy = np.zeros((Ks-1))
stdev_accuracy = np.zeros((Ks-1))
print(mean_accuracy)

for n in range (1,Ks):
    winningtree = DecisionTreeClassifier(criterion= 'entropy', max_depth = n) # here the max_depth will be changed
    winningtree.fit(X_train, Y_train)
    yhat_predictionTree = winningtree.predict(X_val)
    print(yhat_predictionTree)
    print(Y_val)
    print("Decisiontree accuracy: ", metrics.accuracy_score(Y_val, yhat_predictionTree))
    mean_accuracy[n-1] = metrics.accuracy_score(Y_val, yhat_predictionTree)
    stdev_accuracy[n-1] = np.std(predictionTree == Y_val)/np.sqrt(predictionTree.shape[0])

mean_accuracy
stdev_accuracy

import matplotlib.pyplot as plt
from importlib import reload
plt=reload(plt)
plt.plot(range(1,Ks), mean_accuracy, 'blue')
plt.fill_between(range(1,Ks), mean_accuracy - 1 * stdev_accuracy, mean_accuracy + 1 * stdev_accuracy, alpha =0.4)
plt.legend('Accuracy test')
plt.xlabel('Max_depth')
plt.ylabel('Accuracy score')
plt.tight_layout()
plt.show()

#The SVM algorithm offers a choice of kernel functions for performing its processing. Basically, mapping data into a higher dimensional space is called kernelling. The mathematical function used for the transformation is known as the kernel function, and can be of different types, such as:

#1.Linear - 'linear'
#2.Polynomial - 'poly'
#3.Radial basis function (RBF) - 'rbf'
#4.Sigmoid - 'sigmoid'
#5. - ‘precomputed’

from sklearn import svm
classpredict = svm.SVC(kernel='rbf')
classpredict.fit(X_train, Y_train)
yhat_rbf = classpredict.predict(X_val)
yhat_rbf
print("Test set accuracy 'RBF': ", metrics.accuracy_score(Y_val, yhat_rbf))

poly_classpredict = svm.SVC(kernel='poly')
poly_classpredict.fit(X_train, Y_train)
yhat_poly = poly_classpredict.predict(X_val)
yhat_poly
print("Test set accuracy 'Poly': ", metrics.accuracy_score(Y_val, yhat_poly))

sigmoid_classpredict = svm.SVC(kernel='sigmoid')
sigmoid_classpredict.fit(X_train, Y_train)
yhat_sigmoid = sigmoid_classpredict.predict(X_val)
yhat_sigmoid
print("Test set accuracy 'Sigmoid': ", metrics.accuracy_score(Y_val, yhat_sigmoid))

linear_classpredict = svm.SVC(kernel='linear')
linear_classpredict.fit(X_train, Y_train)
yhat_linear = linear_classpredict.predict(X_val)
yhat_linear
print("Test set accuracy 'Linear': ", metrics.accuracy_score(Y_val, yhat_linear))

#--------Predict using Linear model------------
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

LR = LogisticRegression(C=0.01,solver ='liblinear').fit(X_train, Y_train)
LR
yhat_LR = LR.predict(X_val)
yhat_LR
print("Test set accuracy 'LR': ", metrics.accuracy_score(Y_val, yhat_LR))

#------------------------------------------------
#------MODEL EVALUATING USING TEST SET-----------
from sklearn.metrics import f1_score
# for f1_score please set the average parameter to 'micro'
from sklearn.metrics import log_loss



def jaccard_index(predictions, true):
    if (len(predictions) == len(true)):
        intersect = 0;
        for x,y in zip(predictions, true):
            if (x == y):
                intersect += 1
        return intersect / (len(predictions) + len(true) - intersect)
    else:
        return -1
