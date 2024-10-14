Diabetes Prediction Model
This project implements a machine learning model to predict whether a person has diabetes based on several health-related metrics. It uses a Support Vector Machine (SVM) classifier trained on a dataset from the Pima Indians Diabetes Database.

Table of Contents
Overview
Requirements
Dataset
Installation
Usage
Model Evaluation
Contributing
License
Overview
The goal of this project is to predict diabetes in individuals using various health measurements. The model is trained using features such as age, glucose level, blood pressure, skin thickness, insulin level, BMI, diabetes pedigree function, and number of pregnancies.

Requirements
To run this code, you need to have the following libraries installed:

numpy
pandas
scikit-learn
You can install them using pip:

bash
Copy code
pip install numpy pandas scikit-learn
Dataset
The dataset used in this project is the Pima Indians Diabetes Database, which is available from various sources online. The dataset should be placed in the path specified in the code (D:\Projects\Diabetes Prediction\diabetes.csv).

Dataset Features
Pregnancies: Number of times pregnant
Glucose: Plasma glucose concentration
Blood Pressure: Diastolic blood pressure (mm Hg)
Skin Thickness: Triceps skin fold thickness (mm)
Insulin: 2-Hour serum insulin (mu U/ml)
BMI: Body mass index (weight in kg/(height in m)^2)
Diabetes Pedigree Function: A function that scores the likelihood of diabetes based on family history
Age: Age (years)
Outcome: Class variable (0 or 1) indicating absence or presence of diabetes
Installation
Clone the repository or download the script.
Ensure the dataset is correctly placed in the specified path.
Install the required libraries using the command mentioned above.
Usage
Run the Python script.
Enter the required health metrics when prompted.
The model will output whether the person has diabetes or not.
Example Input
mathematica
Copy code
Enter age: 30
Enter glucose level: 85
Enter blood pressure: 66
Enter skin thickness: 29
Enter insulin level: 0
Enter BMI: 26.6
Enter diabetes pedigree function: 0.351
Enter number of times pregnant: 1
Example Output
Copy code
Person does not have diabetes
Model Evaluation
The model’s performance can be evaluated using accuracy scores for both training and test datasets, which are printed in the console. The data is split into 90% training and 10% testing to assess how well the model generalizes.

Contributing
Feel free to fork this repository and submit pull requests for any improvements or bug fixes.

License
This project is open-source and available under the MIT License.
