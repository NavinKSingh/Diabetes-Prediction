import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score

diabetes_data = pd.read_csv('D:\Projects\Diabetes Prediction\diabetes.csv')

diabetes_data.head()

diabetes_data.shape

diabetes_data.size

diabetes_data.describe()

diabetes_data['Outcome'].value_counts()

diabetes_data.groupby('Outcome').mean()

x = diabetes_data.drop(columns='Outcome',axis=1)
y = diabetes_data['Outcome']

print(x)
print(y)

scaler = StandardScaler()

standard_data = scaler.fit_transform(x)

x = standard_data
y =diabetes_data['Outcome']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1,stratify=y,random_state=1)

x.shape,x_train.shape,x_test.shape

classifier = svm.SVC(kernel='linear')

classifier.fit(x_train,y_train)

x_train_prediction = classifier.predict(x_train)
training_data_accuracy =accuracy_score(x_train_prediction,y_train)

print("Accracy Score of training data is: ",training_data_accuracy)

x_test_prediction = classifier.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction,y_test)

print("Accuracy Score of test data is: ",test_data_accuracy)

age = int(input("Enter age: "))
glucose = float(input("Enter glucose level: "))
blood_pressure = float(input("Enter blood pressure: "))
skin_thickness = float(input("Enter skin thickness: "))
insulin = float(input("Enter insulin level: "))
bmi = float(input("Enter BMI: "))
diabetes_pedigree = float(input("Enter diabetes pedigree function: "))
pregnancies = int(input("Enter number of times pregnant: "))
input_data = (age, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, pregnancies)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
std_data = scaler.transform(input_data_reshaped)
prediction = classifier.predict(std_data)
if prediction[0] == 1:
    print("Person has diabetes")
else:
    print("Person does not have diabetes")
