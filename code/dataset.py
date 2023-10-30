#* dataset.py
#*
#* ANLY 555 Fall 2023
#* Project <Data Science Python Toolbox>
#*
#* Due on: Sep. 19, 2023
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
        
        ## user entered file name; user entered data type, must be one of the following: "quantitative", "qualitative", "timeseries", "text". 
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
        
        
#         data_col = input('Please enter the column name(s) you want to manipulate from this list %s, separate with \",\".'% self.colnms)
#         user_list = data_col.split(',')
        user_list = self.df.keys()
        ## the user defined data extracted from input csv as a dictionary
        self.data = {k:v for k,v in self.df.items() if k in user_list}
        
    def __load(self):
        """!
        The load function will prompt the user to enter the name of a file 
        – assumedly which stores a data set to load

        @param filename String: the file path of the file containing the data
        @return Return the testing message showing the method is invoked
        """
        filename = input('Please enter the file name with csv extension, e.g. \"file.csv\":')
        data_type = input('Please enter the data type. Choose from the following: quantitative, qualitative, timeseries, or text: ')
        if data_type not in ["quantitative", "qualitative", "timeseries", "text"]: 
            raise Exception('Please enter a valid data type choosing from the following: quantitative, qualitative, timeseries, text')
        
        return filename, data_type
    

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
    
    
    def _DataSet__clean_col(self, colnm):
        """!
        try convert str to float if possible
        """
        try:
            self.data[colnm] = [float(s) if s!=self.naChar else s for s in self.data[colnm]]
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

        
    
    def clean(self, window_size = 3):
        """!
        Time series median filter with optional parameters which determine the filter size.

        @param: window_size is the filter size to smooth the time series data, default 3
        """
        for colnm in self.ts_colnm:
            # colnm from str to float if possible
            is_num_col = self._DataSet__clean_col(colnm)
            # if float column, then fill missing values by mean
            if is_num_col:
                smooth_ls, n = [], len(self.data[colnm])
                for i in range(n):
                    window = self.data[colnm][max(0, i-window_size):i+1]
                    smooth_ls.append(np.median(window))
                self.data[colnm + '_cleaned'] = smooth_ls
        return
        

    
    
    def explore(self, colnm):
        """
        explore dataset with line plot and histogram

        @param col String: column name to be plotted

        """
        ## time column
        time_index = np.arange(self.startingtime, self.startingtime+len(self.data[colnm]), self.timeDelta).tolist()

        # time series scatter plot
        plt.scatter(time_index, self.data[colnm],marker='o', s=10, label='Data Points') 
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title('Scatter Plot the smoothed data for '+colnm)
        plt.legend()
        plt.grid(True)
        plt.show()
        

        # time series histogram
        plt.hist(self.data[colnm])
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
        
    
    def clean(self):
        """!
        Text data clean to remove stopwords
        """
        for colnm in self.text_colnm:  
            self.data[colnm + 'cleaned'] = []
            for text in self.data[colnm]:
                words = word_tokenize(text)
                stop_words = set(stopwords.words('english'))
                filtered_words = [word for word in words if word.lower() not in stop_words]
                self.data[colnm + 'cleaned'].append(filtered_words)
        return 
        
    
    def explore(self, colnm):
        """!
        explore dataset with plots...
        """
        # word clound
        text = ' '.join(self.data[colnm])
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

        for colnm in self.colnms:
            ## colnm from str to float if possible
            is_num_col = self._DataSet__clean_col(colnm)
            ## if float column, then fill missing values by mean
            if is_num_col:
                ## if each element in the column should be consider as NA
                index = [i!=self.naChar for i in self.data[colnm]]

                ## compute mean after filtering out NA
                fill_value = sum(x for i,x in zip(index, self.data[colnm]) if i)/sum(index)
                
                name = colnm + 'cleaned'
                ## replace NA with mean
                self.data[name] = [x if i else fill_value for i,x in zip(index, self.data[colnm])] 

        return
        
    
    def explore(self,method,column):
        """!
        explore dataset with box plot and pie chart given a column
        @param method String: histogram ot boxplot
        @param column String: the name of the column
        """
        if method == "boxplot":
            # Create a boxplot
            plt.boxplot(self.data[column])

            # Add a title and labels
            plt.title('Boxplot of Sample Data')
            plt.xlabel('Category')
            plt.ylabel('Values')

            # Display the boxplot
            plt.show()
        
        elif method == "histogram":
            plt.hist(self.data[column], bins=20, edgecolor='black')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.title(column + 'Histogram')
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
        
    def ordered(self, colnm):
        """!
        sort the qualitative data...
        other descriptions to be filled
        """
        return sorted(self.data[colnm])
    
    def clean(self):
        """!
        clean QualDataSet including trim the special characters...
        filling missing value with mode.
        """
        for colnm in self.data:
            self._DataSet__clean_col(colnm)  

        for colnm in self.qual_colnms:
            ## compute the most frequent value
            mode_value = max(set(self.data[colnm]), key=self.data[colnm].count)
            self.data[colnm + 'cleaned'] = [i if i!= self.naChar else mode_value for i in self.data[colnm]] 
        return 
        
    
    def explore(self, colmn, n = 5):
        """
        explore dataset with pie chart with top n most popular values

        @param colmn String: the column we want to explore
        @param n int: number of most frequent values we want to see on the plot. 
        """
        x = self.data[colmn]

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
        plt.title('Top 10 Categories and Others')

        ## Display the pie chart
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.show()

            
        return