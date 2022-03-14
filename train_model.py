import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib

customer_data = pd.read_csv('Churn_Modelling.csv')
#print(customer_data.head(10))

#Drop irrelevant features
dataset = customer_data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

#Change categorical to numerical
geography_df = pd.get_dummies(dataset.Geography).iloc[:, 1:]
gender_df = pd.get_dummies(dataset.Gender).iloc[:,1:]

#drop original columns
dataset =  dataset.drop(['Geography', 'Gender'], axis=1)


#Add geography and gender columns to the dataset df
dataset = pd.concat([dataset,geography_df,gender_df], axis=1)
#print(dataset.head(5))

#Get train X and y
X =  dataset.drop(['Exited'], axis=1)
y = dataset['Exited']


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

print(X_test.head(1))
print(y_test.head(1))

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


classifier = RandomForestClassifier(n_estimators=200, random_state=0) 
classifier.fit(X_train, y_train)

print(X_train.columns.values.tolist())

#predict
predictions = classifier.predict(X_test)


print(classification_report(y_test,predictions ))
print(accuracy_score(y_test, predictions))

#Feature Extraction and Importance

feat_importances = pd.Series(classifier.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()


#Save model
joblib.dump(classifier, 'model.joblib')

