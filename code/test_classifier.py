#=====================================================================
# Testing script for Deliverable 3: Classifier
#=====================================================================

#=====================================================================
# Testing Classifier Class 
#=====================================================================
from classifier_algorithm import (Classifier, simpleKNNClassifier, _heap)
from experiment import Experiment
from dataset import (DataSet)

def ImportData():
    print("==============================================================") 
    print("Import iris training dataset from iris_train")
    print("Enter test value sequentially: ../data/iris_train.csv quantitative")
    iris_training = DataSet() 
    iris_training.clean() # convert numbers to numbers
    attr = ['petal_length', 'petal_width', 'sepal_length', 'sepal_width']
    x_train = {col: iris_training.data[col] for col in attr}
    y_train = iris_training.data['species']
    print("Training data without labels: \n")
    print(x_train)
    print("Training true labels: \n")
    print(y_train)
    
    print("==============================================================") 
    print("Import iris testing dataset from iris_test")
    print("Enter test value sequentially: ../data/iris_test.csv quantitative")
    iris_testing = DataSet() 
    iris_testing.clean() # convert numbers to numbers
    attr = ['petal_length', 'petal_width', 'sepal_length', 'sepal_width']
    x_test = {col: iris_testing.data[col] for col in attr}
    y_test = iris_testing.data['species']
    print("Testing data without labels: \n")
    print(x_test)
    print("Testing true labels: \n")
    print(y_test)

def ClassifierTest():

    print("==============================================================")   
    print("Testing the Classifier Class \n")
    iris_classifier = Classifier()
    print("Testing the init method \n")
    print("Initial data: should be \{\} \n")
    print(iris_classifier.data)
    print("Initial true label: should be \[\] \n")
    print(iris_classifier.label)
    print("Initial testData: should be \{\} \n")
    print(iris_classifier.testData)
    print("Initial testlabel: should be \[\] \n")
    print(iris_classifier.testlabel)
    print("Initial k: should be 10 \n")
    print(iris_classifier.k)
    print("==============================================================") 
    print("Testing the train method \n")

    print("Enter test value sequentially: ../data/iris_train.csv quantitative") # import data
    iris_training = DataSet() 
    iris_training.clean() # convert numbers to numbers
    attr = ['petal_length', 'petal_width', 'sepal_length', 'sepal_width']
    x_train = {col: iris_training.data[col] for col in attr}
    y_train = iris_training.data['species']

    print("Enter test value sequentially: ../data/iris_test.csv quantitative")
    iris_testing = DataSet() 
    iris_testing.clean() # convert numbers to numbers
    attr = ['petal_length', 'petal_width', 'sepal_length', 'sepal_width']
    x_test = {col: iris_testing.data[col] for col in attr}
    y_test = iris_testing.data['species']

    iris_classifier.train(x_train, y_train)
    print(iris_classifier.data) # expected: {'petal_length': [1.4,
#   1.3,
#   1.7,
#   5.0,
#   5.1,
#   4.4,
#   1.7,
#   1.5,
#   4.7,...
    print(iris_classifier.label) # expected: ['setosa',
#  'setosa',
#  'setosa',
#  'virginica',
#  'virginica',
#  'versicolor',
#  'setosa',...

    print("==============================================================")
    print("Testing the test method.")
    iris_classifier.test(x_test, y_test, k=3)
    print("Test the testData expected: {'petal_length': [4.1,4.9,1.7,1.3,5.2,5.6,5.0,1.7,1.4,...")
    print(iris_classifier.testData) 
    print("Test the testlabel expected: ['versicolor','versicolor','setosa','setosa','virginica','virginica',...]")
    print(iris_classifier.testlabel) 
    print("Test the k: should be 3 \n")
    print(iris_classifier.k)

def heapClassTest():
    print("==============================================================")
    print("Testing the heap helper class.")
    print("Obtain the top 3 closest points from the following list [(8.727272727272727, 0), (6.0606060606060606, 0), (3.878787878787879, 0), (2.1818181818181817, 0), (0.9696969696969697, 0), (0.24242424242424243, 1), (0.0, 1), (0.24242424242424243, 1), (0.9696969696969697, 1), (2.1818181818181817, 1)]")
    x = [(8.727272727272727, 0), (6.0606060606060606, 0), (3.878787878787879, 0), (2.1818181818181817, 0), 
     (0.9696969696969697, 0), (0.24242424242424243, 1), (0.0, 1), (0.24242424242424243, 1), 
     (0.9696969696969697, 1), (2.1818181818181817, 1)]
    k = 3
    temp = _heap(x)
    print("Expected results: [(0.24242424242424243, 1), (0.0, 1), (0.24242424242424243, 1)]" )
    print(temp.heap_sort(k))


def simpleKNNClassifierTest():
    import numpy as np
    print("==============================================================")
    print("Testing the simpleKNNClassifier class.")
    simplekNNtest = simpleKNNClassifier()


    print("Enter test value sequentially: ../data/iris_train.csv quantitative") # import data
    iris_training = DataSet() 
    iris_training.clean() # convert numbers to numbers
    attr = ['petal_length', 'petal_width', 'sepal_length', 'sepal_width']
    x_train = {col: iris_training.data[col] for col in attr}
    y_train = iris_training.data['species']

    print("Enter test value sequentially: ../data/iris_test.csv quantitative")
    iris_testing = DataSet() 
    iris_testing.clean() # convert numbers to numbers
    attr = ['petal_length', 'petal_width', 'sepal_length', 'sepal_width']
    x_test = {col: iris_testing.data[col] for col in attr}
    y_test = iris_testing.data['species']

    print("Testing the test method.")
    simplekNNtest.fit(x_train, y_train)
    print("The predicted labels are: \n")
    print(simplekNNtest.test(x_test, 3))

def ExperimentTest():
    print("==============================================================")
    print("Testing the Experiment class.")
    
    print("Enter test value sequentially: ../data/iris_train.csv quantitative") # import data
    iris_training = DataSet() 
    iris_training.clean() # convert numbers to numbers
    attr = ['petal_length', 'petal_width', 'sepal_length', 'sepal_width']
    x_train = {col: iris_training.data[col] for col in attr}
    y_train =iris_training.data['species']

    print("Enter test value sequentially: ../data/iris_test.csv quantitative")
    iris_testing = DataSet() 
    iris_testing.clean() # convert numbers to numbers
    attr = ['petal_length', 'petal_width', 'sepal_length', 'sepal_width']
    x_test = {col: iris_testing.data[col] for col in attr}
    y_test = iris_testing.data['species']

    test = Experiment(x_train, y_train)
    print("Testing the crossValidation method.")
    print("The average score of 2-fold of top 2 closest points of the training data is ")
    print(test.crossValidation(2, 2))
    print("The score of being correct is ")
    x = simpleKNNClassifier()
    x.fit(x_train, y_train)
    y_pred = x.test(x_test, 3)
    print(test.score(y_pred,y_train))
    print("The confusionMatrix is ")
    print(test.confusionMatrix(y_pred,y_train))



def main():
    ImportData()
    ClassifierTest()
    heapClassTest()
    simpleKNNClassifierTest()
    ExperimentTest()


if __name__=="__main__":
    main()