# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and read the data frame using pandas.
2. Calculate the null values present in the dataset and apply label encoder. 
3. Determine test and training data set and apply decison tree regression in dataset.
4. calculate Mean square error,data prediction and r2.


## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Nandhini s
RegisterNumber: 212222220028 
*/
import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
x.head()
y=data[["Salary"]]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
```

## Output:
Data.head():

![Screenshot 2023-11-07 203127](https://github.com/nandhu6523/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/123856724/59296cf9-810d-4bba-bdcf-68f839abd105)

Data.info():


   ![Screenshot 2023-11-07 203135](https://github.com/nandhu6523/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/123856724/4c2d95f4-7a1f-4ec8-af73-f8a88bd0da09)

isnull() and sum():

![Screenshot 2023-11-07 204057](https://github.com/nandhu6523/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/123856724/7c757e87-e9e7-41bb-8494-eb2c17e2c4b4)

Data.head() for salary:

![Screenshot 2023-11-07 203321](https://github.com/nandhu6523/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/123856724/5bfd2c23-f863-4d0f-b9f6-02d0f49883e2)

MSE value:
 
![Screenshot 2023-11-07 203839](https://github.com/nandhu6523/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/123856724/5165d2c1-5f54-41d4-8130-552c1c88e4a9)
 
r2 value:
 ![Screenshot 2023-11-07 203835](https://github.com/nandhu6523/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/123856724/52857809-e98f-4d17-b610-8f7fac527efc)


Data prediction:

![Screenshot 2023-11-07 203905](https://github.com/nandhu6523/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/123856724/ed825842-b689-4c24-97c8-a5e739023e9f)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
