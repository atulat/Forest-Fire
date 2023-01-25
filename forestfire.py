import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt

# reading the dataset
df = pd.read_csv("/Users/atulat/Documents/SVM/forestfires.csv")

# Boxplot for the temperature
plt.figure(figsize=(12,5),facecolor="yellow")
plt.boxplot(df['temp'],vert = False)
plt.ylabel("Temperature",size=20)
plt.title("Boxplot",size=20)

# Boxplot for the Wind
plt.figure(figsize=(12,5),facecolor="pink")
plt.boxplot(df['wind'],vert = False)
plt.ylabel("Wind",size=20)
plt.title("Boxplot",size=20)

# Boxplot for the Relative Humidity
plt.figure(figsize=(12,5),facecolor="lightblue")
plt.boxplot(df['RH'],vert = False)
plt.ylabel("Relative Humidity",size=20)
plt.title("Boxplot",size=20)

# Duff Moisture - Violin Plot
plt.figure(figsize=(8,6),facecolor="lightgreen")
plt.violinplot(df["DMC"],vert = False,)
plt.xlabel("hp")

# Corelation between the Himidity and the temperature
plt.figure(figsize=(8,6),)
plt.scatter(df["RH"],df["temp"])
plt.xlabel("Relative Humidity", size=20)
plt.ylabel("Temperature", size=20)

# Jitter plot for temperature month wise
plt.figure(figsize=(8,7))
sns.stripplot(x='month', y='temp',data=df)
plt.ylabel('tip ($)')
plt.show()

# Boxplot for Relative Humidity month wise
plt.figure(figsize=(8,7))
sns.boxplot(x='month', y='RH',data=df)
plt.ylabel('tip ($)')
plt.show()

df = df.drop('month', axis =1)

df = df.drop('day', axis =1)

X = df.iloc[:,:-1]
Y = df.iloc[:,-1]

X = pd.get_dummies(X)

# splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.3,random_state=15)

# print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# Model1 with Linear
clf1 = SVC(kernel="linear",gamma=0.0001)
clf1.fit(X_train , y_train)
y_pred = clf1.predict(X_test)
acc1 = accuracy_score(y_test, y_pred) * 100
print("Linear Accuracy =", acc1)
confusion_matrix(y_test, y_pred)

# Model2 with RBF
clf2 = SVC(C= 1, gamma = 0.001,kernel="rbf")
clf2.fit(X_train , y_train)
y_pred_train= clf2.predict(X_test)
acc2 = accuracy_score(y_test, y_pred_train) * 100
print("RBF Accuracy =", acc2)

# Model3 with Polynomial
clf3 = SVC(C= 1, gamma = 0.001,kernel="poly")
clf3.fit(X_train , y_train)
y_pred_train= clf3.predict(X_test)
acc3 = accuracy_score(y_test, y_pred_train) * 100
print("Polynomial Accuracy =", acc3)

# Model 4 with sigmoid
clf4 = SVC(kernel="sigmoid")
clf4.fit(X_train , y_train)
y_pred_test = clf4.predict(X_test)
acc4 = accuracy_score(y_test, y_pred_test) * 100
print("Sigmoid Accuracy =", acc4)

# Out of all the Models, Model 1 with Linear has higher accuracy.

pred_diff= pd.DataFrame({'actual': Y,
                         'predicted_values': clf1.predict(X)})

print(pred_diff.sample(5))
