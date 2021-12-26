import csv
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import plotly.plotly as py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier
import seaborn as sns
from sklearn import tree

numIterations = 5

sns.set(color_codes=True)

df = pd.read_csv("poker-hand.csv")

# Split into train and test
train, test = train_test_split(df, test_size = 0.20, random_state=1)

# Train set
label = 'hand'
train_y = train[label]
train_x = train[[x for x in train.columns if label not in x]]
# Test/Validation set
test_y = test[label]
test_x = test[[x for x in test.columns if label not in x]]


# ##Run experiment on n_estimators
# training_accuracy = []
# validation_accuracy = []
# test_accuracy = []
# times = []
# n_estimators =np.arange(1,100, 5)
#
# for n in n_estimators:
#     trainSum = 0
#     testSum = 0
#     crossSum = 0
#     startTime = time.time()
#     print n
#
#     for iteration in range(0, numIterations):
#         clf = AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=7, min_samples_split=100),
#                                  n_estimators=n, random_state=1)
#         clf.fit(train_x, train_y)
#         trainSum += accuracy_score(train_y, clf.predict(train_x))
#         crossSum += cross_val_score(clf, train_x, train_y, cv=10).mean()
#         testSum += accuracy_score(test_y, clf.predict(test_x))
#
#     elapsedTime = time.time() - startTime
#     times.append(round(elapsedTime, 3))
#     trainAvg = round(trainSum / numIterations, 3)
#     crossAvg = round(crossSum / numIterations, 3)
#     testAvg = round(testSum / numIterations, 3)
#
#     training_accuracy.append(trainAvg)
#     validation_accuracy.append(crossAvg)
#     test_accuracy.append(testAvg)
#
# fig = plt.figure()
# plt.style.use('ggplot')
# line1, = plt.plot(n_estimators, (training_accuracy), 'r', label="Training Accuracy")
# line2, = plt.plot(n_estimators, (validation_accuracy), 'b', label="Cross Validation Score")
# line1, = plt.plot(n_estimators, (test_accuracy), 'g', label="Testing Accuracy")
# plt.xlabel('Number of Estimators')
# plt.ylabel('Accuracy')
# plt.legend(loc='best')
# plt.title('Number of Estimators versus Accuracy (hand)')
# fig.savefig('figures/ph/boosting/poker-hand_boosting_estimator.png')
# plt.close(fig)
#
# thefile = open('data/ph/boosting/poker-hand_boosting_estimator.txt', 'w')
#
# thefile.write("Number of Estimators:\n")
# for item in n_estimators:
#     thefile.write("%s," % item)
#
# thefile.write("\nTraining Accuracy:\n")
# for item in training_accuracy:
#     thefile.write("%s," % item)
#
# thefile.write("\nValidation Accuracy:\n")
# for item in validation_accuracy:
#     thefile.write("%s," % item)
#
# thefile.write("\nTest Accuracy:\n")
# for item in test_accuracy:
#     thefile.write("%s," % item)
#
# thefile.write("\nTime:\n")
# for item in times:
#     thefile.write("%s," % item)


## After finding the right estimator number, experiment on training set size

training_accuracy = []
validation_accuracy = []
test_accuracy = []
times = []
training_size = [0.1, 0.3, 0.5, 0.7, 0.9]

for n in training_size:
  trainSum = 0
  testSum = 0
  crossSum = 0
  startTime = time.time()
  print (n)

  for iteration in range(0, numIterations):
      clf = AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=7, min_samples_split=100),
                               n_estimators=60, random_state=1)
      temp_train, _ = train_test_split(train, test_size= 1 - n, random_state=1)
      percent_train_y = temp_train[label]
      percent_train_x = temp_train[[x for x in train.columns if label not in x]]
      clf.fit(percent_train_x, percent_train_y)
      trainSum += accuracy_score(percent_train_y, clf.predict(percent_train_x))
      crossSum += cross_val_score(clf, percent_train_x, percent_train_y, cv=10).mean()
      testSum += accuracy_score(test_y, clf.predict(test_x))

  elapsedTime = time.time() - startTime
  times.append(round(elapsedTime, 3))
  trainAvg = round(trainSum / numIterations, 3)
  crossAvg = round(crossSum / numIterations, 3)
  testAvg = round(testSum / numIterations, 3)

  training_accuracy.append(trainAvg)
  validation_accuracy.append(crossAvg)
  test_accuracy.append(testAvg)

fig = plt.figure()
plt.style.use('ggplot')
line1, = plt.plot(training_size, (training_accuracy), 'r', label="Training Accuracy")
line2, = plt.plot(training_size, (validation_accuracy), 'b', label="Cross Validation Score")
line1, = plt.plot(training_size, (test_accuracy), 'g', label="Testing Accuracy")
plt.xlabel('Training Set Size (%)')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.title('Training Size versus Accuracy (hand)')
fig.savefig('figures/ph/boosting/poker-hand_boosting_trainingSize.png')
plt.close(fig)

thefile = open('data/ph/boosting/poker-hand_boosting_trainingSize.txt', 'w')

thefile.write("Training Size:\n")
for item in training_size:
    thefile.write("%s," % item)

thefile.write("\nTraining Accuracy:\n")
for item in training_accuracy:
    thefile.write("%s," % item)

thefile.write("\nValidation Accuracy:\n")
for item in validation_accuracy:
    thefile.write("%s," % item)

thefile.write("\nTest Accuracy:\n")
for item in test_accuracy:
    thefile.write("%s," % item)

thefile.write("\nTime:\n")
for item in times:
    thefile.write("%s," % item)
