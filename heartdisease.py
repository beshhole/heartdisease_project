import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 12)

data = pd.read_csv('E:/Besha/belajar/pandas exercise/exercise 2/heart.csv')

# age = person's age'
# sex = gender
# cp: chest pain experienced (1= typical angina, 2= atypical angina, 3= non-anginal pain, 4= asymptomatic)
# trestbps: person's resting blood pressure (mm Hg on admission to the hospital)
# chol: person's cholesterol measurement in mg/dl
# fbs: person's fasting blood sugar (> 120 mg/dl, 1 = true; 0 = false)
# restecg: Resting electrocardiographic measurement (0 = normal, 1 = having ST-T wave abnormality, 2 = showing probable or definite left ventricular hypertrophy by Estes' criteria)
# thalach: person's maximum heart rate achieved
# exang: Exercise induced angina (1 = yes; 0 = no)
# oldpeak: ST depression induced by exercise relative to rest ('ST' relates to positions on the ECG plot. See more here)
# slope: the slope of the peak exercise ST segment (Value 1: upsloping, Value 2: flat, Value 3: downsloping)
# ca: The number of major vessels (0-3)
# thal: A blood disorder called thalassemia (3 = normal; 6 = fixed defect; 7 = reversable defect)
# target: Heart disease (0 = no, 1 = yes)

sns.heatmap(data.corr(), cmap='coolwarm', annot=True)
plt.show()

sns.countplot(x='target', data=data)
plt.show()

print(pd.crosstab(data['age'], data['target']))
sns.countplot(x='age', data=data, hue='target')
plt.legend(['healthy', 'heart disease'])
plt.title('heart disease distribution by age')
plt.show()

sns.countplot(x='sex', data=data, hue='target')
plt.legend(['healthy', 'heart disease'])
plt.title('heart disease distribution by sex')
plt.show()

x = data.drop('target', axis=1)
y = data['target']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3)

logreg = LogisticRegression()
logreg.fit(xtrain, ytrain)

predicts = logreg.predict(xtest)

print(confusion_matrix(ytest, predicts))
print(classification_report(ytest, predicts))

print(data.describe())
print(data.head())
print(data.info())
