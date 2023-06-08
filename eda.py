import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy import stats
from scipy.stats import zscore

# this pipeline accepts a pandas dataframe and the name of the categorical target variable in the provided dataframe
# this pipeline was intended for prepping data for classification tasks, so numeric targets are not accepted at this time
# optional: provide categorical variable frequency cut off (currently at 5% or higher frequency in target population), whether to merge or exclude outliers 
# (default is exclude) and whether to use plotly or seaborn for figures (default is seaborn)

class EDA:
    def __init__(self, df, target, outlier_threshold=0.05, merge_outliers=False, use_plotly=False):
        
        # original dataset, pre-processed
        self.original_data = df
        self.original_data = self.original_data[self.original_data.columns[:len(df.columns)]]
        # dataframe that will be updated post outlier cleanup / normalization
        self.df = df
        self.target = target
        self.outlier_threshold = outlier_threshold
        self.merge_outliers = merge_outliers
        self.use_plotly = use_plotly
        self.pair_plot_cols=self.original_data.select_dtypes(include=np.number).columns.tolist()
        self.outliers = {}
        
        # run exploratory data analysis pipeline
        self.apply_eda()
        
    def pair_plot(self):
        #numerical = self.original_data.select_dtypes(include=np.number).columns.tolist()
        sns.pairplot(self.original_data[self.pair_plot_cols],hue=self.target)
        
    def update_pair_plot(self, variable_list):
        self.pair_plot_cols = variable_list
        return variable_list
    
    def eda_numerical(self, column):
        
        print(f"Descriptive Statistics for {column}:\n{self.df[column].describe()}\n")
        print(f"Number of missing values in {column}: {self.df[column].isnull().sum()}\n")
        
        # z score the data and return the absolute value
        self.df[column+'_zscore'] = np.abs(zscore(self.df[column]))

        # identify outliers beyond z 3 in either direction. extreme outliers considered pts plus minus 5
        outliers = self.df[self.df[column+'_zscore'] > 3]
        extreme_outliers = outliers[outliers[column+'_zscore'] > 5]

        # either replace outliers with the median, or remove entirely
        if self.merge_outliers:
            self.df.loc[self.df[column+'_zscore'] > 3, column] = self.df[column].median()
        else:
            self.df = self.df[self.df[column+'_zscore'] <= 3]
            
        # store information about the outliers from this column in the outliers dictionary
        self.outliers[column] = {'variable_type': 'numerical', 
                                       'num_outliers': outliers.shape[0], 
                                       'num_extreme_outliers': extreme_outliers.shape[0], 'outliers':outliers, 'extreme_outliers':extreme_outliers}

        # normalize the data and return to the dataframe
        self.df[column+'_norm'] = (self.df[column] - self.df[column].min()) / (self.df[column].max() - self.df[column].min())
        
        # plot normalized data
        print('plotting normalized data')
        fig, ax = plt.subplots(1, 2, figsize=(14, 7))
        if self.use_plotly:
            fig = px.histogram(self.df, x=column+'_norm')
            fig.show()

            fig = px.box(self.df, y=column+'_norm')
            fig.show()
        else:
            sns.histplot(data=self.df, x=column+'_norm',ax=ax[0])
            sns.boxplot(data=self.df, y=column+'_norm',ax=ax[1])
            plt.tight_layout()
            plt.show()
            
        return self.outliers[column]
    
        
    def eda_categorical(self, var, copy=False):
   
        # frequency analysis
        value_counts = self.df[var].value_counts()
        print(f"Frequency counts for {var}:\n{value_counts}\n")

        # calculate the outlier count threshold based on the percentage (if 100 samples total, as default remove categories with less than 5 entries)
        outlier_count = int(self.outlier_threshold * self.df.shape[0])

        # outlier detection, return list of categories under the threshold
        rare_values = value_counts[value_counts <= outlier_count].index.tolist()
        print(f"Outliers detected in {var}: {rare_values}\n")

        # copy the dataframe
        df_copy = self.df.copy()

        # if True, merge the outliers into 'Other' category, else just remove them
        if self.merge_outliers:
            df_copy[var] = df_copy[var].apply(lambda x: 'Other' if x in rare_values else x)
            print(f"After merging outliers into 'Other', frequency counts for {var}:\n{df_copy[var].value_counts()}\n")
        else:
            df_copy = df_copy[~df_copy[var].isin(rare_values)]
            print(f"After removing outliers, frequency counts for {var}:\n{df_copy[var].value_counts()}\n")
            
        self.outliers[var] = {'variable_type': 'categorical', 
                                       'value_counts':value_counts,
                                       'num_outliers': outlier_count, 
                                       'rare_values': rare_values}

        # plot the data with seaborn
        fig, ax = plt.subplots(1, 2, figsize=(14, 7))

        sns.countplot(x=var, data=df_copy, ax=ax[0])
        ax[0].set_title(f'Distribution of {var}')
        ax[0].tick_params(axis='x', rotation=90)

        sns.countplot(x=self.target, hue=var, data=df_copy, ax=ax[1])
        ax[1].set_title(f'{var} vs {self.target}')
        ax[1].tick_params(axis='x', rotation=90)

        plt.tight_layout()
        plt.show()

        if copy:
            print(f"Copy of dataframe with changes:\n{df_copy}")
        else:
            self.df = df_copy
        
        return self.outliers[var]

    def gen_run_label(self,cat=True):
        if cat:
            print('#' * 20)
            print('CATEGORICAL VARIABLE ANALYSIS')
            print('#'* 20, '\n')
        else:
            print('#' * 20)
            print('NUMERICAL VARIABLE ANALYSIS')
            print('#'* 20, '\n')

    def apply_eda(self):
        
        categorical_vars = self.df.select_dtypes(include=['category','int','object']).columns.tolist()
        numerical_vars = self.df.select_dtypes(include=np.number).columns.tolist()
        
        if len(categorical_vars) > 1:
            self.gen_run_label()
            for var in categorical_vars:
                if var == self.target:
                    continue
                else:
                    self.eda_categorical(var)
        else:
            print('No categorical data available\n')

        if len(numerical_vars) > 0:
            self.gen_run_label(False)
            for var in numerical_vars:
                if var == self.target:
                    continue
                else:
                    self.eda_numerical(var)
            
            self.pair_plot()
        else:
            print('\nNo numerical data available')