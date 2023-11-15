#=====================================================================
# Testing script for Deliverable 2: DataSet
#=====================================================================

#=====================================================================
# Testing DataSet Class 
#=====================================================================
from dataset import (DataSet, QuantDataSet, QualDataSet,
                                        TextDataSet, TimeSeriesDataSet)

def DataSetTests():
    print("DataSet _load and readFromCSV function test")
    print("==============================================================")
    print("Quantitative dataset")
    print("Enter test value sequentially: ../data/Sales_Transactions_Dataset_Weekly.csv quantitative")
    transaction = DataSet() 
    print("Check column values") # should be {'W0': ['11','7','7',...'0']}
    print({k:v[:5] for k,v in transaction.data.items()})
    print("Data after clean: \n")
    transaction.clean()
    print({k:v[:5] for k,v in transaction.data.items()})
    print("Now call DataSet.explore()...")
    transaction.explore()
    print("\n\n")
    print("==============================================================")
    print("Qualitative dataset")
    print("Enter test value sequentially: ../data/multiple_choice_responses.csv qualitative")
    print("Check kwarg column name index (colnmidx), colnmidx=1 starting read from line 2 instead of line 1 (colnmidx=0)")
    multiple_choice = DataSet(colnmidx=1) 
    print("Check 'What is your age (# years)?' values") # should be {'What is your age (# years)?': ['22-24','40-44','55-59',...]}
    print({k:v[:5] for k,v in multiple_choice.data.items()})
    print("Data after clean: \n")
    multiple_choice.clean()
    print({k:v[:5] for k,v in multiple_choice.data.items()})
    print("Now call DataSet.explore()...")
    multiple_choice.explore()
    print("\n\n")
    print("==============================================================")
    print("Time series dataset")
    print("Enter test value sequentially: ../data/mitbih_test.csv timeseries")
    print("Check kwarg column name index (colnmidx), colnmidx=-1 with default column name starting from 1")
    mitbih = DataSet(colnmidx=-1, naChar='') 
    print("Check column \'1\' values") # should be {{'1': ['1.000000000000000000e+00','9.084249138832092285e-01','7.300884723663330078e-01'...]}
    print({k:v[:5] for k,v in mitbih.data.items()})
    print("Data after clean: \n")
    mitbih.clean()
    print({k:v[:5] for k,v in mitbih.data.items()})
    print("Now call DataSet.explore()...")
    print(mitbih.explore())
    print("\n\n")
    print("==============================================================")
    print("Text dataset")
    print("Enter test value sequentially: ../data/yelp.csv text text")
    yelp = DataSet() 
    print("Check text values") # should be 
    print({k:v[:5] for k,v in yelp.data.items()})    
    print("Data after clean: \n")
    yelp.clean()
    print({k:v[:5] for k,v in yelp.data.items()})
    print("Now call DataSet.explore()...")
    yelp.explore()
    print("\n\n")
    print("==============================================================")
    
    

def QuantDataSetTests():
    print("Quantitative dataset")
    print("Enter test value sequentially: ../data/Sales_Transactions_Dataset_Weekly.csv quantitative")
    transaction_sub = QuantDataSet(quant_colnms=['W0','W1','W2','W3','W4'])
    print("Quant data head after clean: \n")
    transaction_sub.clean()
    print({k:v[:5] for k,v in transaction_sub.data.items()})
    print("QuantDataSet.explore():")
    transaction_sub.explore("boxplot","W0")
    transaction_sub.explore("histogram","W1")
    print("\n\n")
    
def QualDataSetTests():
    print("Qualitative dataset")
    print("Enter test value sequentially: ../data/multiple_choice_responses.csv qualitative")
    multiple_choice_sub = QualDataSet()
    print("QualDataSet.ordered(): ")
    multiple_choice_sub.ordered(colnm = 'Q1')
    print("Qual data after clean: \n")
    multiple_choice_sub.clean()
    print({k:v[:5] for k,v in multiple_choice_sub.data.items()})
    print("QuanlDataSet.explore():")
    multiple_choice_sub.explore(n=3, colmn="Q2")
    print("\n\n")
    
def TextDataSetTests():
    print("Text dataset")
    print("Enter test value sequentially: ../data/yelp.csv text")
    yelp_sub = TextDataSet(newLine_colnm = '\n', newLine = ',review,') ## check text data delim
    print("Text data after clean: \n")
    yelp_sub.clean()
    print({k:v[:5] for k,v in yelp_sub.data.items()})
    print("TextDataSet.explore():")
    yelp_sub.explore(colnm="funny")
    print("\n\n")
    
def TimeSeriesDataSetTests():
    print("Time series dataset")
    print("Enter test value sequentially: ../data/mitbih_test.csv timeseries")
    mitbih_sub = TimeSeriesDataSet(time=1, start=0, naChar='', colnmidx = -1, 
                                  timeseries_colnames = [str(i+1) for i in range(5)]) 
    print("Time series data after clean: \n")
    mitbih_sub.clean(window_size=300)
    print({k:v[:5] for k,v in mitbih_sub.data.items()})
    print("TimeSeriesDataSet.explore():")
    mitbih_sub.explore(colnm="1_cleaned")
    print("\n\n")

def main():
    DataSetTests()
    QuantDataSetTests()
    QualDataSetTests()
    TextDataSetTests()
    TimeSeriesDataSetTests()

if __name__=="__main__":
    main()