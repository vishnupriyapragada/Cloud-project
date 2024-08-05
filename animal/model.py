import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df=pd.read_csv('Animal_detect - Sheet1.csv')
X = df[['width','height']].values
y = df['class'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
rf_model = RandomForestClassifier(n_estimators=12, criterion = "entropy")
rf_model.fit(X_train, y_train)
rf_train_accuracy = accuracy_score(y_train, rf_model.predict(X_train))
print("Random Forest Training Accuracy:", rf_train_accuracy)
pickle.dump(rf_model,open('model.pkl','wb'))