#!/usr/bin/env python

import pandas as pd
import numpy as np
from data_cleaning_processing import clean_and_process
from data_prep import data_preparation
import model1 as m1
import model2 as m2
import model3 as m3
from ensemble import aggregate_prediction

train, test = clean_and_process('data/train.csv')
X_train, X_test, y_train, y_test, X_final_test, y_final_test = data_preparation(train, test)
df1 = pd.concat([X_train, pd.DataFrame(y_train).reset_index(drop=True)], axis=1)
df2 = pd.concat([X_test, pd.DataFrame(y_test).reset_index(drop=True)], axis=1)
final_train = pd.concat([df1, df2], axis=0)
X_final_train = final_train.iloc[:,:-1]
y_final_train = final_train['Class']
print('\nModel 1:\n')
best_params = m1.hyperparameter_tuning(X_train, y_train)
m1.prediction_results(X_train, y_train, X_test, y_test, best_params)
fsi_1, y_proba_1 = m1.prediction_results(X_final_train, y_final_train, X_final_test, y_final_test, best_params, ensemble=True, charts=True)
print('\nModel 2:\n')
best_params = m2.hyperparameter_tuning(X_train, y_train)
m2.prediction_results(X_train, y_train, X_test, y_test, best_params)
fsi_2, y_proba_2 = m2.prediction_results(X_final_train, y_final_train, X_final_test, y_final_test, best_params, ensemble=True, charts=True)
print('\nModel 3:\n')
best_params = m3.hyperparameter_tuning(X_train, y_train)
m3.prediction_results(X_train, y_train, X_test, y_test, best_params)
fsi_3, y_proba_3 = m3.prediction_results(X_final_train, y_final_train, X_final_test, y_final_test, best_params, ensemble=True, charts=True)
print('Ensemble Results:\n')
agg_proba = np.array([y_proba_1, y_proba_2, y_proba_3]).T
agg_fsi = [fsi_1, fsi_2, fsi_3]
aggregate_prediction(agg_proba, agg_fsi, y_final_test, charts=True)