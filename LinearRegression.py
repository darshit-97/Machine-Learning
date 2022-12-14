import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
from matplotlib import style
import pickle

data = pd.read_csv("Datasets/student/student-mat.csv", sep=";")
# Since our data is separated by semicolons we need to do sep=";"

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

predict = "G3"

X = np.array(data.drop([predict], 1))  # Features
y = np.array(data[predict])  # Labels

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)


# Training model for multiple times to get the best score
'''best = 0
for _ in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)  # acc = accuracy
    print(acc)

    if acc > best:
        best = acc
        with open("studentModel.pickle", "wb") as f:  # Saving our model using pickle
            pickle.dump(linear, f)'''

# Load model
pickle_in = open("studentModel.pickle", "rb")
linear = pickle.load(pickle_in)

print('Coeficient: \n', linear.coef_)  # slope value
print('Intercept: \n', linear.intercept_)  # intercept

predictions = linear.predict(x_test)  # gets list of all predictions
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])


# Plotting our model to graph
p = 'absences'
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()
