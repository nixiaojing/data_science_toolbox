#=====================================================================
# Testing script for Deliverable 5: HeterogeneousDataSet
#=====================================================================

#=====================================================================
# Testing HeterogeneousDataSet Class 
#=====================================================================
from dataset import (DataSet, QuantDataSet, QualDataSet,
                                        TextDataSet, TimeSeriesDataSet, HeterogeneousDataSet)

def HeterogeneousDataSetTests():
    print("==============================================================")
    print("HeterogeneousDataSet __init__ function test")
    print("==============================================================")
    mixed_data = HeterogeneousDataSet(data_file = "../data/mixed_data.csv", dtype_file = "../data/mixed_datatype.csv")
    print("Check the loaded data: \n")
    print({k:v[:5] for k,v in mixed_data.data.items()})
    print("Check the loaded data type: \n")
    print({k:v[:5] for k,v in mixed_data.dtypes.items()})

    print("==============================================================")
    print("HeterogeneousDataSet clean function test")
    print("==============================================================")
    mixed_data.clean()
    print("Quantitative dataset after cleaning: \n")
    print({k:v[:5] for k,v in mixed_data.quant_col.quant_data.items()})
    print("Qualitative dataset after cleaning: \n")
    print({k:v[:5] for k,v in mixed_data.quali_col.qual_data.items()})
    print("Time series dataset after cleaning: \n")
    print({k:v[:5] for k,v in mixed_data.time_col.ts_data.items()})
    print("Text dataset after cleaning: \n")
    print({k:v[:5] for k,v in mixed_data.text_col.text_data.items()})
    print("Combined dataset after cleaning: \n")
    print({k:v[:5] for k,v in mixed_data.data.items()})

    print("==============================================================")
    print("HeterogeneousDataSet explore function test")
    print("==============================================================")
    mixed_data.explore()

    print("==============================================================")
    print("HeterogeneousDataSet select function test")
    print("==============================================================")
    print("Quantitative dataset after cleaning: \n")
    quant = mixed_data.select("quantitative")
    print({k:v[:5] for k,v in quant.items()})
    print("Qualitative dataset after cleaning: \n")
    quali = mixed_data.select("qualitative")
    print({k:v[:5] for k,v in quali.items()})
    print("Time series dataset after cleaning: \n")
    ts = mixed_data.select("timeseries")
    print({k:v[:5] for k,v in ts.items()})
    print("Text dataset after cleaning: \n")
    text = mixed_data.select("text")
    print({k:v[:5] for k,v in text.items()})


def main():
    HeterogeneousDataSetTests()

if __name__=="__main__":
    main()
    