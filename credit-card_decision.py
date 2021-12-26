import csv
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn import tree
import seaborn as sns
 
numIterations = 5
 
sns.set(color_codes=True)
 
df = pd.read_csv("credit-card.csv")
 
# Split into train and test
train, test = train_test_split(df, test_size = 0.20, random_state=1)
 
# Train set
label = 'default payment next month'
train_y = train[label]
train_x = train[[x for x in train.columns if label not in x]]
# Test/Validation set
test_y = test[label]
test_x = test[[x for x in test.columns if label not in x]]


#  RUN EXPERIMENTS ON:  n_features
training_accuracy = []
validation_accuracy = []
test_accuracy = []
times = []
n_features = range(1, 24, 2)
 
for n in n_features:
  trainSum = 0
  testSum = 0
  crossSum = 0
  startTime = time.time()
  print(n)

  for iteration in range(0, numIterations):
      clf = tree.DecisionTreeClassifier(max_features=n, random_state=1)
      clf.fit(train_x, train_y)
      trainSum += accuracy_score(train_y, clf.predict(train_x))
      crossSum += cross_val_score(clf, train_x, train_y, cv=10).mean()
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
line1, = plt.plot(n_features, (training_accuracy), 'r', label="Training Accuracy")
line2, = plt.plot(n_features, (validation_accuracy), 'b', label="Cross Validation Score")
line1, = plt.plot(n_features, (test_accuracy), 'g', label="Testing Accuracy")
plt.xlabel('Max Features')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.title('Max Features versus Accuracy (default payment next month)')
fig.savefig('figures/cc/decision/credit-card_decision_features.png')
plt.close(fig)

thefile = open('data/cc/decision/credit-card_decision_features.txt', 'w')

thefile.write("Max Features:\n")
for item in n_features:
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


#  RUN EXPERIMENTS ON:  max_depth
training_accuracy = []
validation_accuracy = []
test_accuracy = []
times = []
n_depth = range(1, 24, 2)
 
for n in n_depth:
  trainSum = 0
  testSum = 0
  crossSum = 0
  startTime = time.time()
  print(n)
 
  for iteration in range(0, numIterations):
      clf = tree.DecisionTreeClassifier(max_depth=n, random_state=1)
      clf.fit(train_x, train_y)
      trainSum += accuracy_score(train_y, clf.predict(train_x))
      crossSum += cross_val_score(clf, train_x, train_y, cv=10).mean()
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
line1, = plt.plot(n_depth, (training_accuracy), 'r', label="Training Accuracy")
line2, = plt.plot(n_depth, (validation_accuracy), 'b', label="Cross Validation Score")
line1, = plt.plot(n_depth, (test_accuracy), 'g', label="Testing Accuracy")
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.title('Max Depth versus Accuracy (default payment next month)')
fig.savefig('figures/cc/decision/credit-card_decision_depth.png')
plt.close(fig)
 
thefile = open('data/cc/decision/credit-card_decision_depthtxt', 'w')
 
thefile.write("Max Depth:\n")
for item in n_depth:
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


#  RUN EXPERIMENTS ON:  n_samples
training_accuracy = []
validation_accuracy = []
test_accuracy = []
times = []
n_samples = [10,20,30,40,50,100,200,300,400,600,800,1000]
 
for n in n_samples:
  trainSum = 0
  testSum = 0
  crossSum = 0
  startTime = time.time()
  print(n)
 
  for iteration in range(0, numIterations):
      clf = tree.DecisionTreeClassifier(min_samples_split=n, random_state=1)
      clf.fit(train_x, train_y)
      trainSum += accuracy_score(train_y, clf.predict(train_x))
      crossSum += cross_val_score(clf, train_x, train_y, cv=10).mean()
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
line1, = plt.plot(n_samples, (training_accuracy), 'r', label="Training Accuracy")
line2, = plt.plot(n_samples, (validation_accuracy), 'b', label="Cross Validation Score")
line1, = plt.plot(n_samples, (test_accuracy), 'g', label="Testing Accuracy")
plt.xlabel('Min Num of Samples / Split')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.title('Min Num of Samples / Split versus Accuracy (default payment next month)')
fig.savefig('figures/cc/decision/credit-card_decision_samplespng')
plt.close(fig)
 
thefile = open('data/cc/decision/credit-card_decision_samples.txt', 'w')
 
thefile.write("Min Num of Samples / Split:\n")
for item in n_samples:
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



#  RUN EXPERIMENTS For Learning Curve
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
        # Define the classifier
        hiddens = tuple(3 * [32])
        clf = tree.DecisionTreeClassifier(max_depth=5, min_samples_split=600, random_state=1)

        temp_train, _ = train_test_split(train, test_size= 1 - n, random_state=1)

        # Train set
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
plt.xlabel('Training Size')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.title('Training Size versus Accuracy (default payment next month)')
fig.savefig('figures/cc/decision/credit-card_decision_tsize.png')
plt.close(fig)
 
thefile = open('data/cc/decision/credit-card_decision_tsize.txt', 'w')
 
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