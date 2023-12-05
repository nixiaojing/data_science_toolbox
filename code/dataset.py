#* dataset.py
#*
#* ANLY 555 Fall 2023
#* Project <Data Science Python Toolbox>
#*
#* Due on: Dec. 5, 2023
#* Author(s): Xiaojing Ni
#*
#*
#* In accordance with the class policies and Georgetown's
#* Honor Code, I certify that, with the exception of the
#* class resources and those items noted below, I have neither
#* given nor received any assistance on this project other than
#* the TAs, professor, textbook and teammates.
#*

import sys
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import matplotlib.pyplot as plt
import os
import wordcloud
# from datetime import datetime, timedelta
from wordcloud import WordCloud    
from nltk.corpus import stopwords


class DataSet():
    """!
    Define a dataset with methods to manipulate, store or retrieve data or related info
    
    """
    def __init__(self, **kwargs):
        """!
        Initiate the class
        
        @param filename String: the file that contains data
        @param delim String: the deliminator in the input data. Default ','
        @param newLine String: the newline indicator in the input data. Default '\n'
        @param naChar String: the missing values in the input data. Default 'na'
        @param colnmidx int: user defined column name row index, default is "0" indicating the first row is the column names. 
        @param delim_colnm String: the deliminator in the column name line. Default same as delim
        @param newLine_colnm String: the newline indicator in the column name line. Default same as newLine
        Enter -1 for no column name, key will be replaced by number from 1. If this number >0, then will skip all lines on top of the colname line. 

        """

        ## user defined deliminator of the input data, default is ",".
        self._delim = kwargs.get('delim', ',')

        ## user defined new line char of the input data, default is "\n".
        self._newLine = kwargs.get('newLine', '\n')

        ## user defined missing value indicator of the input data, default is "na".
        self.naChar = kwargs.get('naChar', 'na')
        
        ## user defined new line char of the column name line
        self._newLine_colnm = kwargs.get('newLine_colnm', self._newLine)
        
        ## user defind deliminator of the column name line
        self._delim_colnm = kwargs.get('delim_colnm', self._delim)
        
        ## user defined column name row index, default is "0" indicating the first row is the column names.
        self.colnmidx = kwargs.get('colnmidx', 0)

        ## window size for time-series data
        self.window_size = kwargs.get('window_size', 30)

        ## user-provided file name contains data
        self.filename = kwargs.get('data_file', '')

        ## user-provided file name contains data type
        self.datatype = kwargs.get('dtype_file', '')


        
        ## user entered file name; user entered data type, must be one of the following: "quantitative", "qualitative", "timeseries", "text". 
        if self.filename == '' or self.datatype == '':
            self.filename , self.datatype = self.__load()



        ## read data and parse data
        data_str = self.__readFromCSV(self.filename)        
        data_list = data_str.split(self._newLine)
        
        ## get data column names
        if self.colnmidx >= 0:
            ## split line by colnm spliter. First part will be the colnm and second part is the first line data if there is any.
            stack = data_list[self.colnmidx].split(self._newLine_colnm)
            # print(stack)
            ## colunm names
            self.colnms = stack[0].split(self._delim_colnm) # get the col names
            
            ## join all rest of the parts together
            if len(stack) > 1:
                data_list[self.colnmidx] = self._newLine_colnm.join(stack[1:])
        elif self.colnmidx == -1:
            n_column = len(data_list[0].split(self._delim_colnm))
            self.colnms = [str(i) for i in range(1, n_column+1)]
        else:
            print('Invalid input for colnmidx. Looking for a number >= -1 but get %s'% self.colnmidx)
            sys.exit(1)

        ## data as a dictionary with key as column name, value as the list of the data points of the given column name, this is handy when the input data is not a single vector or a mixed type. 
        self.df = {colnm:[] for colnm in self.colnms}

        for row in data_list[self.colnmidx+1:]: # from the next row of the column name, read the data into the dictionary
            elements = row.split(self._delim)
            for elem, colnm in zip(elements, self.colnms):
                self.df[colnm].append(elem) 
        
        
        #  data_col = input('Please enter the column name(s) you want to manipulate from this list %s, separate with \",\".'% self.colnms)
        #  user_list = data_col.split(',')
        user_list = self.df.keys()
        ## the user defined data extracted from input csv as a dictionary
        self.data = {k:v for k,v in self.df.items() if k in user_list}



        ## read and parse the data types file
        dtypes_str = self._DataSet__readFromCSV(self.datatype)
        dtypes_list = dtypes_str.split(self._newLine_colnm)


        ## get data column names
        if self.colnmidx >= 0:
            ## split line by colnm spliter. First part will be the colnm and second part is the first line data if there is any.
            stack_dtypes = dtypes_list[self.colnmidx].split(self._newLine_colnm)
            # print(stack)
            ## colunm names
            self.colnms_dtypes = stack_dtypes[0].split(self._delim_colnm) # get the col names
            
            ## join all rest of the parts together
            if len(stack_dtypes) > 1:
                dtypes_list[self.colnmidx] = self._newLine_colnm.join(stack_dtypes[1:])
        elif self.colnmidx == -1:
            n_column = len(dtypes_list[0].split(self._delim_colnm))
            self.colnms_dtypes = [str(i) for i in range(1, n_column+1)]
        else:
            print('Invalid input for colnmidx. Looking for a number >= -1 but get %s'% self.colnmidx)
            sys.exit(1)
        
        ## type as a dictionary with key as column name, value as the list of the data points of the given column name, this is handy when the input data is not a single vector or a mixed type. 
        self.df_dtypes = {colnm:[] for colnm in self.colnms_dtypes}
        
        elements = dtypes_list[1].split(self._delim)
        # print(self.colnms_dtypes)
        for elem, colnm in zip(elements, self.colnms_dtypes):
            self.df_dtypes[colnm].append(elem) 

        #print(self.df_dtypes)
        user_list = self.df_dtypes.keys()

        ## the user defined data extracted from input csv as a dictionary
        self.dtypes = {k:v for k,v in self.df_dtypes.items() if k in user_list}

        # print(self.dtypes.keys())
        # print(self.data.keys())

        if set(self.dtypes.keys()) != set(self.data.keys()):
            raise Exception("The data types and data columns do not match.")
        
    def __load(self):
        """!
        The load function will prompt the user to enter the name of a file 
        – assumedly which stores a data set to load

        @param datafile String: the file path of the file containing the data
        @param typefile String: the file path of the file containing the data type

        @return data file and type file as string
        """
        datafile = input('Please enter the file name contains the data with csv extension, e.g. \"file.csv\":')
        typefile = input('Please enter the file name contains data type of each column. e.g. \"datatype.csv\": ')
        
        return datafile, typefile
    

    def __readFromCSV(self, filename):
        """!
        Load data from a CSV file

        @param filename String: the file path of the CSV file containing the data
        @return Return the testing message showing the method is invoked
        """
        try:
            # Attempt to open the file for reading
            with open(filename, 'r') as file:
                # Read the data from the file (you may need to parse it)
                data = file.read()
                
            return data
        except TypeError:
            print("Please enter a valid file name with csv extension, e.g. \"file.csv\"")
        except: 
            print("Cannot find the file named \"" + filename + "\", please enter a valid file name.")
        
        
    def clean(self):
        """!
        Convert all missing value to np nan

        """
        for colnm in self.data:
            self._DataSet__clean_col(colnm)
        
        return
    
    
    def _DataSet__clean_col(self, colnm, **kwargs):
        """!
        try convert str to float if possible
        """
        data = kwargs.get('data', self.data)
        try:
            data[colnm] = [float(s) if s!=self.naChar else s for s in self.data[colnm]]
            return True
        except: pass
    
    def explore(self):
        """!
        explore each column with a histogram plot, skip if data type is qual or text

        """
        if self.datatype in ["text"]:
            pass
        else:
            for colnm in self.data:
                plt.hist(self.data[colnm], bins=20, edgecolor='black')
                plt.xlabel('Value')
                plt.ylabel('Frequency')
                plt.title(colnm + ' Histogram')
                plt.show()
        return
        
    
    
class TimeSeriesDataSet(DataSet):
    """!
    Define a time series dataset with methods to manipulate, store or retrieve time-series data or related info

    """    
    def __init__(self, time, start, timeseries_colnames = [],**kwargs):
        """!
        Initiate the class
        
        @param time String: the time interval of the data in second, e.g. 1
        @param start String: the starting point of the series, int, e.g.0
        @param timeseries_colnames String: the column names of time series data
        """
        super().__init__(**kwargs)
        ## the time interval of the data, in second, e.g. 1
        self.timeDelta = time

        ## the starting point of the series, int, e.g. 0
        self.startingtime = start

        # specify the time series columns
        if len(timeseries_colnames) > 0:
            self.ts_colnm = timeseries_colnames
        else: self.ts_colnm = self.colnms
        self.ts_data = {key: value for key, value in self.data.items() if key in self.ts_colnm}

        
    
    def clean(self, window_size = 3):
        """!
        Time series median filter with optional parameters which determine the filter size.

        @param: window_size is the filter size to smooth the time series data, default 3
        """
        for colnm in self.ts_colnm:
            # colnm from str to float if possible
            is_num_col = self._DataSet__clean_col(colnm,data=self.ts_data)
            # if float column, then fill missing values by mean
            if is_num_col:
                smooth_ls, n = [], len(self.ts_data[colnm])
                for i in range(n):
                    window = self.ts_data[colnm][max(0, i-window_size):i+1]
                    smooth_ls.append(np.median(window))
                self.ts_data[colnm + '_cleaned'] = smooth_ls
        return
        

    
    
    def explore(self, colnm):
        """
        explore dataset with line plot and histogram

        @param col String: column name to be plotted

        """
        ## time column
        time_index = np.arange(self.startingtime, self.startingtime+len(self.ts_data[colnm]), self.timeDelta).tolist()

        # time series scatter plot
        plt.scatter(time_index, self.ts_data[colnm],marker='o', s=10, label='Data Points') 
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title('Scatter Plot the smoothed data for '+colnm)
        plt.legend()
        plt.grid(True)
        plt.show()
        

        # time series histogram
        plt.hist(self.ts_data[colnm])
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title(colnm + ' Histogram')
        plt.show()
        
        return 

class TextDataSet(DataSet):
    """!
    Define a text dataset with methods to manipulate, store or retrieve text data or related info
    
    """
    def __init__(self, text_colnames = [],**kwargs):
        """!
        Initiate the class
        
        @param text_colnames list: the selected columns containing text data
        """
        super().__init__(**kwargs)

        ## specify the text columns
        if len(text_colnames) > 0:
            self.text_colnm = text_colnames
        else: self.text_colnm = self.colnms
        self.text_data = {key: value for key, value in self.data.items() if key in self.text_colnm}
    
    def clean(self):
        """!
        Text data clean to remove stopwords
        """
        for colnm in self.text_colnm:  
            self.text_data[colnm + '_cleaned'] = []
            for text in self.text_data[colnm]:
                words = word_tokenize(text)
                stop_words = set(stopwords.words('english'))
                filtered_words = [word for word in words if word.lower() not in stop_words]
                
                self.text_data[colnm + '_cleaned'].append(filtered_words)
        return 
        
    
    def explore(self, colnm):
        """!
        explore text dataset with word cloud plot.
        """
        # word cloud
        flatten_list = sum(self.text_data[colnm], [])
        text = ' '.join(flatten_list)
        # Create a WordCloud object
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

        # Display the word cloud plot
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')  # Hide the axis
        plt.title('Word Cloud Plot')
        plt.show()


        return     
    

class QuantDataSet(DataSet):
    """!
    Define a quantitative dataset with methods to manipulate, store or retrieve quantitative data or related info
    
    """    
    def __init__(self, quant_colnms = [],**kwargs):
        """!
        Initiate the class
        
        @param quant_colnms list: The user defined column names of the quantative data
        """
        super().__init__(**kwargs)

        ## quant_colnms is the user defined column names of the quantative data
        if len(quant_colnms)>0:
            self.quant_colnms = quant_colnms
        else:self.quant_colnms = self.colnms
        
        self.quant_data = {key: value for key, value in self.data.items() if key in self.quant_colnms}
        
    # def clean(self):
    #     """!
    #     Fill in missing values with the mean
    #     """
    #     for colnm in self.data:
    #         self._DataSet__clean_col(colnm)
            
    
    #     for colnm in self.quant_colnms:
    #         fill_value = np.nanmean(self.data[colnm])
    #         name = colnm + 'cleaned'
    #         self.data[name] = [i if i!=self.naChar else fill_value for i in self.data[colnm]] 
    #     return 
    
    
    def clean(self):
        """!
        Fill in missing values with the mean
        """

        for colnm in self.quant_colnms:
            
            ## colnm from str to float if possible
            is_num_col = self._DataSet__clean_col(colnm,data=self.quant_data)
            ## if float column, then fill missing values by mean
            if is_num_col:
                ## if each element in the column should be consider as NA
                index = [i!=self.naChar for i in self.quant_data[colnm]]

                ## compute mean after filtering out NA
                fill_value = sum(x for i,x in zip(index, self.quant_data[colnm]) if i)/sum(index)
                
                name = colnm + '_cleaned'
                ## replace NA with mean
                self.quant_data[name] = [x if i else fill_value for i,x in zip(index, self.quant_data[colnm])] 
        return
        
    
    def explore(self,method,column):
        """!
        explore dataset with box plot and pie chart given a column
        @param method String: histogram ot boxplot
        @param column String: the name of the column
        """
        if method == "boxplot":
            # Create a boxplot
            plt.boxplot(self.quant_data[column])

            # Add a title and labels
            plt.title('Boxplot of ' + column)
            plt.xlabel('Category')
            plt.ylabel('Values')

            # Display the boxplot
            plt.show()
        
        elif method == "histogram":
            plt.hist(self.quant_data[column], bins=20, edgecolor='black')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.title(column + ' Histogram')
            plt.show()

        
        else: print("Please input a valid chart type.")

        return 
    
class QualDataSet(DataSet):
    """!
    Define a qualitative dataset with methods to manipulate, store or retrieve qualitative data or related info
    
    """   
    def __init__(self, qual_colnms = [],**kwargs):
        """!
        Initiate the class
        
        @param qual_colnms list: selected columns with qualitative data
        """
        super().__init__(**kwargs)
        if len(qual_colnms)>0:
            self.qual_colnms = qual_colnms
        else:self.qual_colnms = self.colnms

        self.qual_data = {key: value for key, value in self.data.items() if key in self.qual_colnms}
        
    def ordered(self, colnm):
        """!
        sort the qualitative data...
        other descriptions to be filled
        """
        return sorted(self.qual_data[colnm])
    
    def clean(self):
        """!
        clean QualDataSet including trim the special characters...
        filling missing value with mode.
        """
        for colnm in self.qual_data:
            self._DataSet__clean_col(colnm,data=self.qual_data)  

        for colnm in self.qual_colnms:
            ## compute the most frequent value
            mode_value = max(set(self.qual_data[colnm]), key=self.qual_data[colnm].count)
            self.qual_data[colnm + '_cleaned'] = [i if i!= self.naChar else mode_value for i in self.qual_data[colnm]] 
        return 
        
    
    def explore(self, colmn, n = 5):
        """
        explore dataset with pie chart with top n most popular values

        @param colmn String: the column we want to explore
        @param n int: number of most frequent values we want to see on the plot. 
        """
        x = self.qual_data[colmn]

        if n == 0:
            n = len(set(x))

        ## count element frequency
        dict_count = {}
        for i in x:
            if i not in dict_count:
                dict_count[i] = 0
            dict_count[i] += 1
        
        ## sort elements from the most frequent to least frequent
        stack = sorted([(k,v) for k,v in dict_count.items()], key = lambda x: -x[1])
        
        top_n = {k:v for k,v in stack[:n]}
        top_n['Others'] = sum(i for _,i in stack[n:])
        
        ## Create a pie chart
        plt.figure(figsize=(8, 8))
        plt.pie(top_n.values(), labels=top_n.keys(), autopct='%1.1f%%', startangle=140)

        ## Add a title
        plt.title(f'Top {n} Categories and Others')

        ## Display the pie chart
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.show()

            
        return

class HeterogeneousDataSet(DataSet):
    """!
    Define a dataset with methods to manipulate, store or retrieve data or related info
    
    """
    def __init__(self, **kwargs):
        """!
        Initiate the class, it inherent the parameters from the superclass

        based on the data type, read in the data differently
        """
        super().__init__(**kwargs)
        
        ## column names of qualitative data
        self.qualitative_list = []

        ## column names of quantitative data
        self.quantitative_list = []

        ## column names of time series data
        self.timeseries_list = []

        ## column names of text data
        self.text_list = []

        ## column names of other data
        self.other_list = []

        for key, value_list in self.dtypes.items():
            # Extract the value from the list
            value = value_list[0]

            # Append the key to the corresponding list
            if value == 'qualitative':
                self.qualitative_list.append(key)

            elif value == 'quantitative':
                self.quantitative_list.append(key)
            elif value == 'timeseries':
                self.timeseries_list.append(key)
            elif value == 'text':
                self.text_list.append(key)
            else: self.other_list.append(key)

        if len(self.quantitative_list) > 0: 

            ## QuantDataSet object storing quantitative data
            self.quant_col = QuantDataSet(quant_colnms=self.quantitative_list, data_file = self.filename, dtype_file = self.datatype)
        
        if len(self.qualitative_list) > 0: 
            ## QualDataSet object storing qualitative data
            self.quali_col = QualDataSet(qual_colnms=self.qualitative_list,data_file = self.filename, dtype_file = self.datatype) 
            
        
        if len(self.timeseries_list) > 0: 
            time = int(input('Please enter the time parameter of the timeseries data: e.g.: 1'))
            start = int(input('Please enter the start parameter of the timeseries data: e.g.: 0'))
            ## TimeSeriesDataSet object storing timeseries data
            self.time_col = TimeSeriesDataSet(time=time, start=start, timeseries_colnames=self.timeseries_list,data_file = self.filename, dtype_file = self.datatype)
            
            
        if len(self.text_list) > 0:  
            ## TextDataSet object storing text data     
            self.text_col = TextDataSet(text_colnames=self.text_list,data_file = self.filename, dtype_file = self.datatype) 

        if len(self.other_list) > 0:
            ## QualDataSet object storing other data, normally will be the row index
            self.other_data = QualDataSet(qual_colnms=self.other_list,data_file = self.filename, dtype_file = self.datatype)

        ## a dictionary storing all data
        self.data = {**self.other_data.qual_data, **self.quant_col.quant_data, **self.time_col.ts_data, **self.quali_col.qual_data, **self.text_col.text_data}
        
    def clean(self):
        """!
        Based on the data type, call the corresponding clean method in other classes

        """

        
        if len(self.quantitative_list) > 0: # fill missing with mean
            
            #self.quant_col = QuantDataSet(quant_colnms=self.quantitative_list, data_file = self.filename, dtype_file = self.datatype)
            
            self.quant_col.clean() 
            for colnm in self.quantitative_list:
                
                self.data[colnm + '_cleaned'] = self.quant_col.quant_data[colnm + '_cleaned']
            
        if len(self.qualitative_list) > 0: # fill missing with mode
            #self.quali_col = QualDataSet(qual_colnms=self.qualitative_list,data_file = self.filename, dtype_file = self.datatype) 
            self.quali_col.clean()
            
            for colnm in self.qualitative_list:
                
                self.data[colnm + '_cleaned'] = self.quali_col.qual_data[colnm + '_cleaned']
        
        if len(self.timeseries_list) > 0: # Time series median filter with optional parameters which determine the filter size.
            
            # time = int(input('Please enter the time parameter of the timeseries data: e.g.: 1'))
            # start = int(input('Please enter the start parameter of the timeseries data: e.g.: 0'))
            window_size = int(input('Please enter the window_size parameter of the timeseries data for cleaning: e.g.: 300'))
            # self.time_col = TimeSeriesDataSet(time=time, start=start, timeseries_colnames=self.timeseries_list,data_file = self.filename, dtype_file = self.datatype)
            self.time_col.clean(window_size=window_size)
            for colnm in self.timeseries_list:
                self.data[colnm + '_cleaned'] = self.time_col.ts_data[colnm + '_cleaned']
            
        if len(self.text_list) > 0: # Text data clean to remove stopwords       
                            
            # self.text_col = TextDataSet(text_colnames=self.text_list,data_file = self.filename, dtype_file = self.datatype) 
            self.text_col.clean()
            
            for colnm in self.text_list:
                self.data[colnm + '_cleaned'] = self.text_col.text_data[colnm + '_cleaned']
        
        return
    
    
    def explore(self):
        """!
        explore each column based on data type

        """
        try:
            for column in self.quantitative_list:
                column_cleaned = column+'_cleaned'
                self.quant_col.explore("boxplot",column_cleaned)
                self.quant_col.explore("histogram",column_cleaned)

            for column in self.qualitative_list:
                column_cleaned = column+'_cleaned'
                n = int(input('Please enter number of most frequent values we want to see on the plot: e.g.: 3'))
                self.quali_col.explore(n=n, colmn=column_cleaned)

            for column in self.timeseries_list:
                column_cleaned = column+'_cleaned'
                self.time_col.explore(colnm=column_cleaned)
            
            for column in self.text_list:
                column_cleaned = column+'_cleaned'
                self.text_col.explore(colnm=column_cleaned)
                
        except: print("Something wrong happened. Please run the clean function before explore.")
        
        return
    
    def select(self, datatype):
        """!
        Select a data set from read in data with single data type

        @param datatype string: the data type of desired data set, 

        @return a dictionary of data of the desired data type

        """

        if datatype == "quantitative":
            return self.quant_col.quant_data
        elif datatype == "qualitative":
            return self.quali_col.qual_data
        elif datatype == "timeseries":
            return self.time_col.ts_data
        elif datatype == "text":
            return self.text_col.text_data
        else: raise Exception("Please enter a data type from \"quantitative\", \"qualitative\", \"timeseries\", and \"text\".")

