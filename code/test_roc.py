#=====================================================================
# Testing script for Deliverable 4: ROC curve
#=====================================================================
from classifier_algorithm import (Classifier, simpleKNNClassifier, _heap)
from experiment import Experiment


#=====================================================================
# Testing ROC curve for two classes
#=====================================================================

def MultiClass_Test():
    print("==============================================================") 
    print("Testing ROC curve for multi classes")
    print("==============================================================") 
    print("Input true labels and predicted probabilities. ")
    prob_dict = {"experiment1":{"label":["class2","class1","class1","class2","class3","class3"],"score":{"class1":[0.2,0.3,0.5,0.6,0.1,0.2],"class2":[0.6,0.2,0.2,0.1,0.2,0.2],"class3":[0.2,0.5,0.3,0.3,0.7,0.6]}},"experiment2":{"label":["class2","class1","class1","class2","class3","class3"],"score":{"class1":[0.5,0.2,0.1,0.2,0.8,0.3],"class2":[0.1,0.6,0.6,0.5,0.1,0.1],"class3":[0.4,0.2,0.3,0.3,0.1,0.6]}}}
    print("The example data is ",prob_dict)
    roc_analysis = Experiment(prob_dict = prob_dict)
    print("==============================================================") 
    print("Test attributes: \n")
    print("The input dataset is: ",roc_analysis.prob_dict)
    print("Experiments in the dataset are: ",roc_analysis.experiment) ## same as inputted prob_dict
    print("==============================================================") 
    print("Test methods: \n")
    roc_analysis.plot_roc_curve()
    
def TwoClass_Test():
    print("==============================================================") 
    print("Testing ROC curve for two classes")
    print("==============================================================") 
    print("Input true labels and predicted probabilities. ")
    prob_dict = {"experiment1":{"label":["class2","class1","class1","class2"],"score":{"class1":[0.5,0.8,0.5,0.4],"class2":[0.5,0.2,0.5,0.6]}}, "experiment2":{"label":[1,0,0,1],"score":{"0":[0.9,0.6,0.5,0.1],"1":[0.1,0.4,0.5,0.9]}}}
    print("The example data is ",prob_dict)
    roc_analysis = Experiment(prob_dict = prob_dict)
    print("==============================================================") 
    print("Test attributes: \n")
    print("The input dataset is: ",roc_analysis.prob_dict)
    print("Experiments in the dataset are: ",roc_analysis.experiment) ## same as inputted prob_dict
    print("==============================================================") 
    print("Test methods: \n")
    roc_analysis.plot_roc_curve()


def main():
    TwoClass_Test()
    MultiClass_Test()


if __name__=="__main__":
    main()