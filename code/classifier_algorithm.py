#* classifie_algorithm.py
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

class Classifier():
    '''!
    Provided methods of train and test for Classifier and its' subclasses
    The train method for simplekNNClassifier will have input parameters trainingData and true labels.
    '''
    def __init__(self):
        '''!
        Initialize the class
        '''
        ## the input data without labels
        self.data = {}

        ## the true training label
        self.label = []

        ## the test data without labels
        self.testData = {}

        ## the true testing label
        self.testlabel = []

        ## top k
        self.k = 10

    def train(self, trainingData, trueLabels):
        '''!
        Store the training data and labels
        @param data dictionary: the input training data without labels
        @param label list: the training true labels
        '''
        self.data = trainingData
        self.label = trueLabels

    def test(self, testData, trueLabels, k):
        '''!
        Store the testing data and labels, simplekNNClassifier will rewrite this method
        @param testData dictionary: the input testing data without labels
        @param label list: the testing true labels
        @param k int: top k closest training samples
        '''
        self.testData = testData
        self.testlabel = trueLabels
        self.k = k


class _heap():
    '''!
    Creaet a heap to efficiently search the smallest k distance. It is a maximum heap that we always put the farest 
    point on top. If the new point is closer than the farest one, we pop the top one and push the new one in.
    computation complixity O(KlogN)
    '''
    def __init__(self, data):
        '''!
        @param data list: the input data
        '''
        ## save KNN with largest distance in front.
        self.top_k = []
        ## the input data
        self.data = data
        
    def _heapify(self, node): 
        '''!
        Sort from bottom to top
        '''
        if node == 0: return # stop on root
        if self.top_k[node][0] > self.top_k[(node-1)//2][0]: # if leaf value > root value, swith leaf and root
            self.top_k[node], self.top_k[(node-1)//2] = self.top_k[(node-1)//2], self.top_k[node]
        self._heapify((node-1)//2)  
    
    def heappushpop(self, element, k):
        '''!
        @param element tuple: the new point [(distance, label), ...]
        @param k int: the length of heap
        '''
        if element[0] >= self.top_k[0][0]:
            return
        self.top_k[0] = element
        # sort heap
        for i in range(k):
            self._heapify(i)
        
    def heap_sort(self, k): 
        '''
        sort top k
        '''
        self.top_k = self.data[:k]

        for i in range(k): # sort
            self._heapify(i)
#         print('-'*60)
#         print('init heap', self.top_k)

        for element in self.data[k:]:
            self.heappushpop(element, k)
#         print('-'*60)
#         print('the top 3 closest points', self.top_k)

        return self.top_k
    



    
    
class simpleKNNClassifier(Classifier):
    '''!
    Perform simple kNN classifier
    For each single new point, the computation complixity is O(KlogN). If there are M new points, O(MKlogN)

    To perform a formal computational complexity analysis, let's analyze the three methods in your code: `simpleKNNClassifier`, `Experiment score`, and `Experiment confusion matrix`. 

    Complexity Analysis
    We will focus on the time complexity (T(n)) and space complexity (S(n)) for each method. Since we are assuming the worst-case scenario, we'll analyze the method in terms of the input size 'n,' which could represent the number of data points or features, depending on the context.

    **1. `simpleKNNClassifier` Method:**

    - **Time Complexity (T(n)):** 
    - In the `fit` method, it computes the variance for each column, which takes O(n) time for each column.
    - In the `dist_sqr` method, it computes the squared distance for each data point against every training data point, resulting in O(n^2) time complexity.
    - In the `_pred` method, it sorts the distances, which takes O(K * log(n)) time. It also counts the label frequencies, which takes O(K) time.
    - In the `test` method, if there are M new points, and for each point, it calls `_pred`, the overall time complexity is O(M * K * (log(n) + K)).

    - **Space Complexity (S(n)):**
    - The space complexity is mainly determined by storing the training data and labels. It uses additional space for variables like `res`, `heap`, and `cnt`.

    **Tight-Fit Upperbound Using Big-O Notation:**

    1. `simpleKNNClassifier`:
    - Time Complexity: O(M * K * (log(n) + K))
    - Space Complexity: O(n)

    **Justification:**

    1. The time complexity of `simpleKNNClassifier` depends on the number of new data points (M), the number of neighbors (K), and the number of training data points (n). In the worst case, it needs to calculate distances for each new point against all training points, making the time complexity O(M * K * n).

    '''

    def __init__(self,**kwargs):

        '''!
        Initialize the class
        '''

        super().__init__(**kwargs) # inherent the superclass parameters

        ## variance on each column, used for normalization
        self.scale = {} 

        
    def fit(self, data, label):
        '''!
        provide data and label to fit KNN model.
        @param data dict: the training data without labels
        @param label list: the true label of the training data
        '''
        self.label = label
        for colnm, col in data.items():
            self.scale[colnm] = np.var(col)
            self.data[colnm] = col
                
    def dist_sqr(self, testData):
        '''!
        @param testData dictionary: the test data points without labels {attr: value}

        @return the ordered distance 
        '''
        res = []
        n = len(self.label)
        for i in range(n): ## distance with every single point
            res.append(sum((testData[attr]-self.data[attr][i])**2/self.scale[attr] for attr in testData))
        return res
        
    def _pred(self, testData, k):
        '''!
        predict label for single point using the mod.
        find the k closest training samples and return the mode of the k labels associated from the k closest training samples.

        @param testData dictionary: the test data points without labels
        @param k int: the top k closest points

        @return predicted labels for one point
        '''
        ## compute distance
        dist_list = self.dist_sqr(testData)

        ## select top k close distance
        heap = _heap(list(zip(dist_list, self.label)))
        top_k_label = [label for (_, label) in heap.heap_sort(k)] 
        
        ## count label freq
        cnt = {}
        for label in top_k_label:
            if label not in cnt:
                cnt[label] = 0
            cnt[label] += 1
            
        max_freq = max(cnt.values()) ## mod
        return [k for k,v in cnt.items() if v == max_freq][0]
        
    def test(self, testData, k):    
        '''!
        @param testData dictionary: the test data points without labels
        @param k int: the top k closest points

        @return predicted label for all new points.
        '''
        try: 
            colnms = list(testData.keys())
            n = testData[colnms[0]]
            res = []
            for i in range(len(n)):
                data_slice = {colnm:testData[colnm][i] for colnm in colnms}
                res.append(self._pred(data_slice, k))
            return res
        except: print("Please enter testData as a dictionary and k as an int.")



        


    