import csv
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn import neighbors
from sklearn.model_selection import cross_val_score
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

# training_accuracy = []
# validation_accuracy = []
# test_accuracy = []
# times = []
# k_values = np.power(2, np.arange(1,9))
# # k_values = [1,2,3,4,5,6,7,8,9,10,12,14,16,18,20,30,40,60,70,80,90,100]
 
# for k in k_values:
#   trainSum = 0
#   testSum = 0
#   crossSum = 0
#   startTime = time.time()
#   print(k)
#
#   for iteration in range(0, numIterations):
#       clf = neighbors.KNeighborsClassifier(k)
#       clf.fit(train_x, train_y)
#       trainSum += accuracy_score(train_y, clf.predict(train_x))
#       crossSum += cross_val_score(clf, train_x, train_y, cv=10).mean()
#       testSum += accuracy_score(test_y, clf.predict(test_x))
#
#   elapsedTime = time.time() - startTime
#   times.append(round(elapsedTime, 3))
#   trainAvg = round(trainSum / numIterations, 3)
#   crossAvg = round(crossSum / numIterations, 3)
#   testAvg = round(testSum / numIterations, 3)
#
#   training_accuracy.append(trainAvg)
#   validation_accuracy.append(crossAvg)
#   test_accuracy.append(testAvg)
#
# fig = plt.figure()
# plt.style.use('ggplot')
# line1, = plt.plot(k_values, (training_accuracy), 'r', label="Training Accuracy")
# line2, = plt.plot(k_values, (validation_accuracy), 'b', label="Cross Validation Score")
# line1, = plt.plot(k_values, (test_accuracy), 'g', label="Testing Accuracy")
# plt.xlabel('K Values')
# plt.ylabel('Accuracy')
# plt.legend(loc='best')
# plt.title('K Values versus Accuracy (default payment next month)')
# fig.savefig('figures/cc/knn/credit-card_knn_kvalues.png')
# plt.close(fig)
#
# thefile = open('data/cc/knn/credit-card_knn_kvalues.txt', 'w')
#
# thefile.write("K Values:\n")
# for item in k_values:
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
k = 128
training_size = [0.1, 0.3, 0.5, 0.7, 0.9]

for n in training_size:
  trainSum = 0
  testSum = 0
  crossSum = 0
  startTime = time.time()
  print(n)

  for iteration in range(0, numIterations):
      clf = neighbors.KNeighborsClassifier(k)
      temp_train, _ = train_test_split(train, test_size=1 - n, random_state=1)
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
plt.xlabel('Training size')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.title('Training Size versus Accuracy (default payment next month)')
fig.savefig('figures/cc/knn/credit-card_knn_trainingsize.png')
plt.close(fig)

thefile = open('data/cc/knn/credit-card_knn_trainingsize.txt', 'w')

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


