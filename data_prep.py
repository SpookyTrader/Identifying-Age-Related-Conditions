import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

def data_preparation(train, final_test):
    print('Data preparation for ML in progress.....\n')
    onehot_encoder = OneHotEncoder(drop='first', sparse_output=False)

    onehot_encoder.fit(train[['EJ']])
    X_encoded = onehot_encoder.transform(train[['EJ']])
    X_encoded_df = pd.DataFrame(X_encoded)
    X_encoded_df.columns = onehot_encoder.get_feature_names_out()
    train = pd.concat([train, X_encoded_df], axis=1)
    X = train.drop(['Id', 'EJ', 'Class'], axis=1)
    X.rename(columns={'EJ_B':'EJ'}, inplace=True)
    y = train['Class']

    X_encoded = onehot_encoder.transform(final_test[['EJ']])
    X_encoded_df = pd.DataFrame(X_encoded)
    X_encoded_df.columns = onehot_encoder.get_feature_names_out()
    final_test = pd.concat([final_test, X_encoded_df], axis=1)
    X_final_test = final_test.drop(['Id', 'EJ', 'Class'], axis=1)
    X_final_test.rename(columns={'EJ_B':'EJ'}, inplace=True)
    y_final_test = final_test['Class']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.30, random_state=5)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_final_test = scaler.transform(X_final_test)

    X_train = pd.DataFrame(X_train, columns=X.columns)
    X_test = pd.DataFrame(X_test, columns=X.columns)
    X_final_test = pd.DataFrame(X_final_test, columns=X.columns)

    print('Completed!!!\n')

    return X_train, X_test, y_train, y_test, X_final_test, y_final_test