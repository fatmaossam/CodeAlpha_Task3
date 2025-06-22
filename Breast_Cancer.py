from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix
import pandas as pd
import numpy as np

the_data = load_breast_cancer()
data_frame = pd.DataFrame(data=the_data.data, columns=the_data.feature_names)
data_frame['target'] = the_data.target
unique, counts = np.unique(data_frame['target'], return_counts=True)
print("Target Distribution:", dict(zip(unique, counts)))
x = data_frame.iloc[:,:-1]
y = data_frame.iloc[:,-1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = SVC(kernel='linear',class_weight='balanced')
model.fit(x_train, y_train)
the_predictions = model.predict(x_test)
accuracy = accuracy_score(y_test, the_predictions)*100
print('the accuracy of the model is:', accuracy)
print('the classification report of the model is:\n', classification_report(y_test, the_predictions))
print('the confusion matrix of the model is:\n', confusion_matrix(y_test, the_predictions))
