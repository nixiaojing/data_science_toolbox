#* experiment.py
#*
#* ANLY 555 Fall 2023
#* Project <Data Science Python Toolbox>
#*
#* Due on: Oct. 25, 2023
#* Author(s): Xiaojing Ni
#*
#*
#* In accordance with the class policies and Georgetown's
#* Honor Code, I certify that, with the exception of the
#* class resources and those items noted below, I have neither
#* given nor received any assistance on this project other than
#* the TAs, professor, textbook and teammates.
#*
import numpy as np
from classifier_algorithm import (Classifier, simpleKNNClassifier, _heap)

class Experiment():
    def __init__(self, data, label):
        '''!
        Initialize the class
        '''
        ## the data without label
        self.X = data

        ## the true label
        self.y = label

        ## the number of data points
        self.n = len(label)
    
    def crossValidation(self, k_fold, KNN_k):
        '''!
        @param k_fold int
        @param KNN_k int: the top k points
        @return average score
        '''
        sample_size = self.n//k_fold
        group = list(range(k_fold))*sample_size
        group += [-1]*(self.n - sample_size*k_fold) ## group label by -1 will be removed in X-validation
        np.random.shuffle(group)
        score_list = []
        for i in range(k_fold):
            ## split training and testing
            index = [x == i for x in group]
            train_X, train_y, test_X, test_y = self._split(index)
            
            ## train model
            model = simpleKNNClassifier()
            model.fit(train_X, train_y)
            
            ## predict value
            pred_y = model.test(test_X, KNN_k)
            score_list.append(self.score(pred_y, test_y))
        return sum(score_list)/len(score_list)
            
    def _split(self, boolean_list): ## select True row
        '''!
        split data and label in training and testing
        '''
        train_y = [y for i, y in zip(boolean_list, self.y) if not i]
        test_y  = [y for i, y in zip(boolean_list, self.y) if i]
        
        train_X, test_X = {}, {}
        for k,v in self.X.items():
            train_X[k] = [y for i, y in zip(boolean_list, v) if not i]
            test_X[k] = [y for i, y in zip(boolean_list, v) if i]
        
        return (train_X, train_y, test_X, test_y)        
    
    def score(self, pred_y, real_y):
        '''!
        @param pred_y list: predicted labels
        @param real_y list: true labels
        
        @return the percent of correct predictions in 100/100 

        Complexity Analysis

        **1. `score` Method:**

        - **Time Complexity (T(n)):**
        - The method compares predicted labels to true labels, which takes O(n) time for a list of length 'n'.

        - **Space Complexity (S(n)):**
        - The space complexity is minimal, as it stores the `right_cnt` variable.

        **Tight-Fit Upperbound Using Big-O Notation:**

        1. `score`:
        - Time Complexity: O(n)
        - Space Complexity: O(1)

        **Justification:**

        1. The `score` method has a straightforward linear time complexity for comparing predicted labels to true labels.

        The space complexity depends on the data size and the variables used in each method. Please note that these complexities can vary based on the specific use case and how the methods are called in practice.
        '''
        right_cnt = sum(a == b for (a,b) in zip(pred_y, real_y))
        return right_cnt/len(real_y)
        
    def confusionMatrix(self, pred_y, real_y, order = None):
        '''!
        Return confusion matrix and row, column names. For the confusion matrix, row is for predict y and column is for true y. If the desired order is not given, the confusion matrix is with random name order. 

        @param pred_y list: predicted labels
        @param real_y list: true labels
        @param order list: the list of the level names with desired order displayed in confusion matrix

        return confusion matrix and row, column names. row is for predict y and column is for true y. example output: 
        ([[6, 4, 4], [7, 6, 4], [4, 5, 5]], ['setosa', 'versicolor', 'virginica'])
        is the confusion matrix as

        |       | setosa | versicolor | virginica  |
        | ----: | :----: | :----: | :---- |
        | setosa    | 6     | 4     |   4    |
        | versicolor  | 7   | 6   | 5  |
        | virginica  | 4   | 5   | 5  |

        Complexity Analysis

        **1. `confusionMatrix` Method:**

        - **Time Complexity (T(n)):**
        - This method calculates the confusion matrix, iterating through predicted and true labels, which has a time complexity of O(n).

        - **Space Complexity (S(n)):**
        - The space complexity depends on the size of the data and labels.
        - It also uses space for the `mtx` matrix and `order` list.

        **Tight-Fit Upperbound Using Big-O Notation:**

        1. `confusionMatrix`:
        - Time Complexity: O(n)
        - Space Complexity: O(n)

        **Justification:**

        1. The `confusionMatrix` method iterates through the predicted and true labels to create the confusion matrix, resulting in a linear time complexity.

        The space complexity depends on the data size and the variables used in each method. Please note that these complexities can vary based on the specific use case and how the methods are called in practice.
        '''
        if not order:
            order = list(set(self.y)) ## assign random order to the matrix
        nm_index_dict = {nm:i for i,nm in enumerate(order)}
        m = len(order)
        mtx = [[0]*m for _ in range(m)] ## row for pred_y, col for real_y
        for yhat, ytrue in zip(pred_y, real_y):
            row_index, col_index = nm_index_dict[yhat], nm_index_dict[ytrue]
            mtx[row_index][col_index] += 1
        return (mtx, order)
