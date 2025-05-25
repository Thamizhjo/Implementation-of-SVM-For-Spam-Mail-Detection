# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the necessary python packages using import statements.

2.Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().

3.Split the dataset using train_test_split.

4.Calculate Y_Pred and accuracy.

5.Print all the outputs.

6.End the Program
## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: HENRIPRASATH S
RegisterNumber:  212223230077
*/
```
```
import chardet
file='spam.csv'
with open (file,'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result

import pandas as pd
data=pd.read_csv("spam.csv",encoding='windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
## ENCODING

![Screenshot 2025-05-25 132246](https://github.com/user-attachments/assets/d34ac87b-fb68-45f8-a0e5-ccf3dcc390d0)


## Head():

![Screenshot 2025-05-25 132256](https://github.com/user-attachments/assets/e6fffc45-897e-46f9-bb6d-f1f904b1fbf2)


## Info():

![Screenshot 2025-05-25 132306](https://github.com/user-attachments/assets/5618900e-130a-4e5f-871b-db6cb433f84a)


## isnul().sum():

![Screenshot 2025-05-25 132313](https://github.com/user-attachments/assets/39e9a76f-8437-4fe3-9f0b-55a2c94517a4)


## Prediction of Y

![Screenshot 2025-05-25 132324](https://github.com/user-attachments/assets/d86fb7d4-76bf-48a6-bedb-b7083c224dea)


## Acuuarcy

![Screenshot 2025-05-25 132333](https://github.com/user-attachments/assets/276d4d69-09d0-43fe-bcfa-8fd2be13a7dc)



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
