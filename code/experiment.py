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
import matplotlib.pyplot as plt

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



class ROCAnalysis:
    def __init__(self, data_dict):
        """
        Initialize the ROCAnalysis object with input data.

        Parameters:
        - data_dict: Dictionary where keys are experiment values are lists lists of label and scores.
        dictionary of dictionary
        data_dict = 
        {"experiment1":
            {"label":[class2,class1,class1,class2,class3,class3],
             "score": ## which n-1 classes, n is the total class number
                {"class1":[0.2,0.3,0.5,0.6,0.1,0.2],
                 "class2":[0.6,0.2,0.2,0.1,0.2,0.2]}
                 }
        "experiment2":
            {"label":[class2,class1,class1,class2,class3,class3],
             "score":
                {"class1":[0.2,0.3,0.5,0.6,0.1,0.2],
                 "class2":[0.6,0.2,0.2,0.1,0.2,0.2]}
                 }
        }
        """
        self.data_dict = data_dict
        self.experiment = list(data_dict.keys())
        self.num_classes = len(set(data_dict[self.experiment[0]]["label"]))
        

    def plot_roc_curve(self):
        """
        Plot ROC curves for binary or multiclass classification.
        """
        if self.num_classes == 2:
            # Binary classification
            self.plot_binary_roc_curve()
        elif self.num_classes > 2:
            # Multiclass classification (one-versus-all ROC curves)
            self.plot_multiclass_roc_curve()
        else:
            print("Error: Invalid number of classes.")

    def plot_binary_roc_curve(self):
        """
        Plot ROC curves for binary classification.
        """
        plt.figure(figsize=(6, 6))

        for i, (experiment, label_score) in enumerate(self.data_dict.items()):
            true_labels = label_score["label"]
            scores_dict = label_score["score"]
            score_class = list(scores_dict.keys())[0]
            scores = list(scores_dict.values())[0]
            binary_labels = [1 if str(true_labels[x])== str(score_class) else 0 for x in range(len(true_labels))]
            fpr, tpr = self.calculate_roc_curve(binary_labels, scores)
            auc_score = self.calculate_auc(fpr, tpr)
            color = self.get_color(i)
            plt.plot(fpr, tpr, color=color, label=str(experiment)+f' Class {score_class} to others, (AUC = {auc_score:.2f})')

        # Plotting the baseline (1:1 line)
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=2, label='Random')
        plt.title("ROC curve")
        plt.xlabel("False Positive Rate (TPR)")
        plt.ylabel("True Positive Rate (FPR)")
        plt.legend()
        plt.show()

    def calculate_roc_curve(self, true_labels, scores):
        """
        Calculate ROC curve for binary classification.

        Parameters:
        - true_labels: True class labels (0 or 1).
        - scores: Predicted scores/probabilities.

        Returns:
        - fpr: False Positive Rate.
        - tpr: True Positive Rate.
        """
        total_positive = sum(true_labels) 
        total_negative = len(true_labels) - total_positive 

        thresholds = list(set(scores)) 
        thresholds.sort(reverse=True) ## using sort to improve efficiency
        fpr, tpr = [0], [0]

        for threshold in thresholds:
            true_positive, false_positive = 0, 0
            for i in range(len(true_labels)):
                predicted_label = 1 if scores[i] >= threshold else 0

                if predicted_label == 1 and true_labels[i] == 1:
                    true_positive += 1
                elif predicted_label == 1 and true_labels[i] == 0:
                    false_positive += 1
            if total_negative and total_positive:
                fpr.append(false_positive / total_negative)
                tpr.append(true_positive / total_positive)
            else: 
                fpr.append(0)
                tpr.append(0)

        return np.array(fpr), np.array(tpr)

    def calculate_auc(self, fpr, tpr):
        """
        Calculate AUC (Area Under the Curve) from ROC curve points.

        Parameters:
        - fpr: False Positive Rate.
        - tpr: True Positive Rate.

        Returns:
        - AUC score.
        """
        auc_score = 0.0
        for i in range(1, len(fpr)):
            auc_score += (fpr[i] - fpr[i - 1]) * (tpr[i] + tpr[i - 1]) / 2.0
        return auc_score

    def get_color(self, index):
        """
        Get a color for plotting based on the index.

        Parameters:
        - index: Index of the class.

        Returns:
        - Color string.
        """
        colors = ['aqua', 'darkorange', 'cornflowerblue', 'green', 'red']
        return colors[index % len(colors)]

    def get_linetype(self, index):
        """
        Get a linetype for plotting based on the index.

        Parameters:
        - index: Index of the experiment.

        Returns:
        - linetype string.
        """
        linetypes = ['solid', 'dotted', 'dashed', 'dashdot']
        return linetypes[index % len(linetypes)]
        



    def plot_multiclass_roc_curve(self):
        """
        Plot one-versus-all ROC curves for multiclass classification.
        """
        plt.figure(figsize=(7, 7))

        for i, (experiment, label_score) in enumerate(self.data_dict.items()):
            true_labels = label_score["label"]
            scores_dict = label_score["score"]
            color = self.get_color(i)
            for j , (classes, scores) in enumerate(scores_dict.items()):
                binary_labels = [1 if str(true_labels[k]) == classes else 0 for k in range(len(true_labels))]

                fpr, tpr = self.calculate_roc_curve(binary_labels, scores)
                auc_score = self.calculate_auc(fpr, tpr)
                linetype = self.get_linetype(j)
                plt.plot(fpr, tpr, color=color, linestyle=linetype,label=str(experiment)+f' Class {classes} to others, (AUC = {auc_score:.2f})')

        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=2, label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve (Multiclass)')
        plt.legend(bbox_to_anchor=(0.95, -0.1),ncol=len(self.experiment),fontsize=6 )
        plt.show()