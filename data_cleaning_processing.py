import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def clean_and_process(file):
    df = pd.read_csv(file)
    print('extracting data...\n')
    print(df.head())

    print('\n\nCleaning and processing data...\n')
    train, test = train_test_split(df, test_size=0.3, random_state=5)
    train.drop_duplicates(keep='first',ignore_index=True, inplace=True)
    test.drop_duplicates(keep='first',ignore_index=True, inplace=True)
    
    train_1 = train.loc[train['Class'] == 1,:].copy()
    train_0 = train.loc[train['Class'] == 0,:].copy()
    train_1.fillna(train_1.select_dtypes('float').median(), inplace=True)
    train_0.fillna(train_0.select_dtypes('float').median(), inplace=True)
    train = pd.concat([train_1, train_0], ignore_index=True)
    
    test_1 = test.loc[test['Class'] == 1,:].copy()
    test_0 = test.loc[test['Class'] == 0,:].copy()
    test_1.fillna(test_1.select_dtypes('float').median(), inplace=True)
    test_0.fillna(test_0.select_dtypes('float').median(), inplace=True)
    test = pd.concat([test_1, test_0], ignore_index=True)

    print('\n\nCleaned and processed datasets:\n')
    print('Train set:\n')
    print(train.info())
    print('\n\nHold-out test set:\n')
    print(test.info())
    print('\n\nDone!!!')
    
    return train, test