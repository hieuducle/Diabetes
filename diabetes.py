import pandas as pd
import numpy as np
from scipy.constants import precision
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from lazypredict.Supervised import LazyClassifier
from math import sqrt


data = pd.read_csv("diabetes.csv")

# result = data.corr()

# plt.figure(figsize=(8, 8))
# sn.histplot(data["Outcome"])
# plt.title("Diabetes distribution")
# plt.savefig("diabetes.jpg")


target = "Outcome"
x = data.drop(target, axis=1)
y = data[target]

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# lazyPredict

# clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
# models,predictions = clf.fit(x_train, x_test, y_train, y_test)
# print(models)

cls = QuadraticDiscriminantAnalysis()
cls.fit(x_train, y_train)
y_predict = cls.predict(x_test)
print(classification_report(y_test, y_predict))

#
# parameters_randomforest = {
#     "n_estimators": [50, 100, 200, 500],
#     "criterion": ["gini", "entropy", "log_loss"],
#     # "max_depth": [None, 5, 10],
#     # "max_features": ["sqrt", "log2"],
# }


# cls = GridSearchCV(QuadraticDiscriminantAnalysis(), param_grid=parameters, scoring="recall", cv=6, verbose=1, n_jobs=6)
# cls.fit(x_train, y_train)
# print(cls.best_score_)
# print(cls.best_params_)

# cm = np.array(confusion_matrix(y_test, y_predict, labels=[0, 1]))
# confusion = pd.DataFrame(cm, index=["Not Diabetic", "Diabetic"], columns=["Not Diabetic", "Diabetic"])
# sn.heatmap(confusion, annot=True)
# plt.savefig("diabetes_prediction.jpg")