import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy import stats
from scipy.stats import zscore

class EDA:
    def __init__(self, df, target, outlier_threshold=0.05, merge_outliers=False, use_plotly=False):
        self.df = df
        self.target = target
        self.outlier_threshold = outlier_threshold
        self.merge_or_remove = merge_or_remove
        self.use_plotly = use_plotly
        self.outliers = {}

        # run exploratory data analysis pipeline
        self.apply_eda()
        
    def eda_numerical(self, column):
        
        print(f"Descriptive Statistics for {var}:\n{df[var].describe()}\n")
        print(f"Number of missing values in {var}: {df[var].isnull().sum()}\n")
        
        # z score the data and return the absolute value
        self.df[column+'_zscore'] = np.abs(zscore(self.df[column]))

        # identify outliers beyond z 3 in either direction. extreme outliers considered pts plus minus 5
        outliers = self.df[self.df[column+'_zscore'] > 3]
        extreme_outliers = outliers[outliers[column+'_zscore'] > 5]

        # either replace outliers with the median, or remove entirely
        if merge_outliers:
            self.df.loc[self.df[column+'_zscore'] > 3, column] = self.df[column].median()
        else:
            self.df = self.df[self.df[column+'_zscore'] <= 3]

        # store information about the outliers from this column in the outliers dictionary
        self.outliers[column] = {'variable_type': 'numerical', 
                                       'num_outliers': outliers.shape[0], 
                                       'num_extreme_outliers': extreme_outliers.shape[0], 'outliers':outliers, 'extreme_outliers':extreme_outliers}

        # plot data
        print('plotting pre-normalized data')
        if use_plotly:
            fig = px.histogram(self.df, x=column)
            fig.show()

            fig = px.box(self.df, y=column)
            fig.show()
        else:
            sns.histplot(data=self.df, x=column)
            plt.show()

            sns.boxplot(data=self.df, y=column)
            plt.show()

        # normalize the data and return to the dataframe
        self.df[column+'_norm'] = (self.df[column] - self.df[column].min()) / (self.df[column].max() - self.df[column].min())
        
        # plot normalized data
        print('plotting normalized data')
        if use_plotly:
            fig = px.histogram(self.df, x=column+'_norm')
            fig.show()

            fig = px.box(self.df, y=column+'_norm')
            fig.show()
        else:
            sns.histplot(data=self.df, x=column+'_norm')
            plt.show()

            sns.boxplot(data=self.df, y=column+'_norm')
            plt.show()
            
        return self.outliers[column]
    
        
    def eda_categorical(self, var, copy=False):
   
        # frequency analysis
        value_counts = df[var].value_counts()
        print(f"Frequency counts for {var}:\n{value_counts}\n")

        # calculate the outlier count threshold based on the percentage (if 100 samples total, as default remove categories with less than 5 entries)
        outlier_count = int(outlier_threshold * df.shape[0])

        # outlier detection, return list of categories under the threshold
        rare_values = value_counts[value_counts <= outlier_count].index.tolist()
        print(f"Outliers detected in {var}: {rare_values}\n")

        # copy the dataframe
        df_copy = df.copy()

        # if True, merge the outliers into 'Other' category, else just remove them
        if merge_outliers:
            df_copy[var] = df_copy[var].apply(lambda x: 'Other' if x in rare_values else x)
            print(f"After merging outliers into 'Other', frequency counts for {var}:\n{df_copy[var].value_counts()}\n")
        else:
            df_copy = df_copy[~df_copy[var].isin(rare_values)]
            print(f"After removing outliers, frequency counts for {var}:\n{df_copy[var].value_counts()}\n")
            
        self.outliers[column] = {'variable_type': 'categorical', 
                                       'value_counts':value_counts,
                                       'num_outliers': outlier_count, 
                                       'rare_values': rare_values}

        # plot the data with seaborn
        fig, ax = plt.subplots(1, 2, figsize=(14, 7))

        sns.countplot(x=var, data=df, ax=ax[0])
        ax[0].set_title(f'Distribution of {var}')
        ax[0].tick_params(axis='x', rotation=90)

        sns.countplot(x=target, hue=var, data=df, ax=ax[1])
        ax[1].set_title(f'{var} vs {target}')
        ax[1].tick_params(axis='x', rotation=90)

        plt.tight_layout()
        plt.show()

        if copy:
            print(f"Copy of dataframe with changes:\n{df_copy}")
        else:
            self.df = df_copy
        
        return self.outliers[column]

    def apply_eda(self):
        # Iterate over the columns in the dataframe
        for var in self.df.columns:
            # Skip the target variable
            if var == self.target:
                continue
            
            # Apply the appropriate EDA method
            if np.issubdtype(self.df[var].dtype, np.number):
                self.eda_numerical(var)
            else:
                self.eda_categorical(var)
