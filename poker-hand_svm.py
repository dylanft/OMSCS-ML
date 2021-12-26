import csv
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import plotly.plotly as py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.model_selection import cross_val_score
from numpy import arange
import seaborn as sns

numIterations = 5

sns.set(color_codes=True)
 
df = pd.read_csv("poker-hand.csv")

# dff = df.sample(10000)

# print(dff.info())
 
# # Split into train and test
train, test = train_test_split(df, test_size = 0.20, random_state=1)
 
# Train set
label = 'hand'
train_y = train[label]
train_x = train[[x for x in train.columns if label not in x]]
# Test/Validation set
test_y = test[label]
test_x = test[[x for x in test.columns if label not in x]]

from sklearn.preprocessing import MinMaxScaler
scaling = MinMaxScaler(feature_range=(-1,1)).fit(train_x)
train_x = scaling.transform(train_x)
test_x = scaling.transform(test_x)

# training_accuracy = []
# validation_accuracy = []
# test_accuracy = []
# times = []
# gamma = [0.0001, 0.001, 0.01, 0.1]
#
# # [poly, linear, rbf]
#
# for n in gamma:
#     trainSum = 0
#     testSum = 0
#     crossSum = 0
#     startTime = time.time()
#     print 'gamma:', n
#
#     for iteration in range(0, numIterations):
#         print(iteration)
#         clf = svm.SVC(kernel='rbf', random_state=1, gamma=n)
#         # clf = svm.LinearSVC(kernel='rbf', random_state=1, gamma=n)
#
#         clf.fit(train_x, train_y)
#
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
# line1, = plt.plot(gamma, (training_accuracy), 'r', label="Training Accuracy")
# line2, = plt.plot(gamma, (validation_accuracy), 'b', label="Cross Validation Score")
# line1, = plt.plot(gamma, (test_accuracy), 'g', label="Testing Accuracy")
# plt.xlabel('Gamma')
# plt.ylabel('Accuracy')
# plt.legend(loc='best')
# plt.title('Gamma versus Accuracy (default payment next month)')
# fig.savefig('figures/ph/svm/poker-hand_svm_gamma_rbf.png')
# plt.close(fig)
#
# thefile = open('data/ph/svm/poker-hand_svm_gamma_rbf.txt', 'w')
#
# thefile.write("Gamma:\n")
# for item in gamma:
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
    print(n)

    for iteration in range(0, numIterations):
        clf = svm.SVC(kernel='linear',gamma=0.01, random_state=1)
        temp_train, _ = train_test_split(train, test_size= 1 - n, random_state=1)

        # Train set
        percent_train_y = temp_train[label]
        percent_train_x = temp_train[[x for x in train.columns if label not in x]]
        scaling = MinMaxScaler(feature_range=(-1,1)).fit(percent_train_x)
        percent_train_x = scaling.transform(percent_train_x)

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
plt.xlabel('Training Size')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.title('Training Size versus Accuracy (default payment next month)')
fig.savefig('figures/ph/svm/poker-hand_svm_tsize.png')
plt.close(fig)

thefile = open('data/ph/svm/poker-hand_svm_tsize.txt', 'w')

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
