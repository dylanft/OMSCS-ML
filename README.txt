Dylan

----------------------------------------

This code was built on Python 2.7
The following packages are required to run the experiments:
- Numpy
- Pandas
- csv
- time
- matplotlib
- plotly
- seaborn
- sklearn

----------------------------------------

In order to run this code you will need to be using a Python 2.7 environment with the list of dependecies having been installed (pip install X).

The two datsets are included as csv files. They are: 'credit-card.csv' and 'poker-hand.csv'

There are 10 python files, and each file can be used on its own to run an experiment for that particular type of supervised learning algorithm.
Experiments begin where training accuracy, cross validation scores, testing accuracy, and training times are reset to empty lists, and the
experiment ends after the lines about writing the results to a png file and a text file. 

There are at least two experiments within each file. The code to create the model complexity plots comes first, for each parameter that was tested
over some range of values. Towards the end of the file is the experiment that generates the learning curves. The 'optimal' parameters from the model
complexity plots are passed into the parameters of the classifier used in the learning curve. These hyper parameters have to be edited manually if
you wish to change them. Text file outputs about the accuracy metrics are saved under the data folder and the model complexity plots for each 
hyper parameter and also the learning curve plots are saved under the figures folder.

In order to run the code you must change directories to this folder and run (example):
python credit-card_svm.py

This will run the python code for the SVM algorithm for the credit card data set and generate the appropriate reports.

The files only run one type of parameter so to adjust this you need to open the file and comment out or uncomment a specific block of code. 
