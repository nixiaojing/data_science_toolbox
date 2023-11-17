# data_science_toolbox

The data science toolbox is written in Python. There are three classes. The toolbox is developed from scratch. No advanced libraries such as Pandas or scikit-learn were used. <br>
<ul>
<li> `Dataset` class is used to load and clean the different types of data. There are four subclasses to manipulate time series data, quantitative data, qualitative data, and text data. 
<li> `Classifier` class is used to perform classification tasks on the given data. 
<li> `Experiment` class is used to perform assessment on classification tasks. The methods such as score and confusion matrix are included. <br>
Please find more detailed documentation on the <a href="https://nixiaojing.github.io/data_science_toolbox/annotated.html">Github Pages</a>. 

## Repository structure

```.
├── README.md
├── code/
├── data/
├── docs/

```

## Description of files


* The `code/` directory contains all code files
    * `dataset.py`: DataSet Class has 5 main member methods. 
    	* a. The load function will prompt the user to enter the name of a file - assumedly which stores a data set to load.
		* b. The clean method should ?clean? data according to category of data as follows:
			* i. Quant data should fill in missing values with the mean
			* ii. Qual data should fill in missing values with the median or mode
			* iii. Time series information should run a median filter with optional parameters which determine the filter size.
			* iv. Text data should remove stop words (and feel free to stem and / or lemmatize).
		* c. The explore method creates visualizations of the data. 
    * `classifier_algorithm.py`: 
    	* a. simpleKNNClassifier test method 
			* i. The train method for simplekNNClassifier has input parameters trainingData and true labels. 
			* ii. The test method has parameters testData and k, and finds the k closest training samples and return the mode of the k labels associated from the k closest training samples. The predicted labels will be stored in a member attribute and will also be returned.
    * `experiment.py`: The Experiment Constructor will take as input one data set , labels, and a list of classifiers. Each will be stored in a member attribute.
		* a. crossValidation method takes as input kFolds. This method will perform k-fold crossvalidation, and for each fold will train all classifiers (on the training folds), and test all classifiers on the testing folds.
		* b. The score method computes the accuracy of each classifier and present the result as a table.
		* c. The confusionMatrix method computes and display a confusion matrix for each classifier.
		* d. ROC related methods result in plot ROC curve for two-class and multi-class classification problems.
    * `test_classifier.py`: test the functionality of classifier class.
    * `test_dataset.py`: test the functionality of dataset class.
    * `test_roc.py`: test the functionality of ROC related methods in Experiment class 
    * `test_structure.py`: test the functionality of the toolbox structure.


