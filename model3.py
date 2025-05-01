import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, roc_auc_score, classification_report,accuracy_score, balanced_accuracy_score, make_scorer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

def balanced_logloss(y_test, y_predict_proba):
    
    y_test = y_test.to_numpy()
    N0 = np.sum(1 - y_test)
    N1 = np.sum(y_test)
    
    p1 = np.clip(y_predict_proba, 1e-15, 1 - 1e-15)
    p0 = 1 - p1
    
    log_loss_0 = -np.sum((1 - y_test) * np.log(p0)) / N0
    log_loss_1 = -np.sum(y_test * np.log(p1)) / N1
    
    return (log_loss_0 + log_loss_1)/2
    
def hyperparameter_tuning(X_train, y_train):

    model = LGBMClassifier(verbose=-1, random_state=5, scale_pos_weight=8)

    params = {'max_depth': np.arange(3,15,1),
              # 'scale_pos_weight': np.arange(4,8,0.5),
              'learning_rate': np.arange(0.01,0.2,0.01),
              'subsample': np.arange(0.5, 1.0, 0.01),
              'colsample_bytree': np.arange(0.5, 1.0, 0.01),
              'colsample_bynode': np.arange(0.5, 1.0, 0.01),
              'n_estimators': np.arange(100,1000,50)}

    bal_acc = make_scorer(balanced_accuracy_score)

    best_tree = RandomizedSearchCV(estimator=model, param_distributions=params, scoring=bal_acc, cv=5, n_iter=100, n_jobs=-1, verbose=True, random_state=5)
    best_tree.fit(X_train, y_train)
    
    best_params = best_tree.best_params_
    best_score = best_tree.best_score_
    
    print('\nBest parameters for Model 3:', best_params)
    print('\nBest score for Model 3:', best_score)

    return best_params

def prediction_results(X_train, y_train, X_test, y_test, best_params, ensemble=False, charts=False):
    
    model = LGBMClassifier(verbose=-1, random_state=5, scale_pos_weight=8)
    model.set_params(**best_params)
    model.fit(X_train, y_train)

    y_predict = model.predict(X_test)
    print('Prediction on test data:', y_predict) 

    confusion = metrics.confusion_matrix(y_test, y_predict)
    print('\nConfusion matrix:\n', confusion)
    if charts:
        confusion_df = pd.DataFrame(confusion, columns=np.unique(y_test), index = np.unique(y_test))
        plt.figure(figsize = (3,3))
        plt.rcParams.update({'font.size': 15})
        sns.heatmap(confusion_df, cmap = 'Blues', annot=True, fmt='g', square=True, linewidths=.5, cbar=False)
        plt.xlabel('Predicted')
        plt.ylabel('Expected')
        plt.tight_layout()
        plt.savefig('images/confusion_matrix_model3.png')
        plt.close()

    y_predict_proba = model.predict_proba(X_test)[:,1]
    print('\nPredicted probabilities:',y_predict_proba)
    print('\nBalanced Log Loss:{}'.format(balanced_logloss(y_test, y_predict_proba)))

    bal_accuracy = balanced_accuracy_score(y_test,y_predict)
    print('\nBalanced_Accuracy_score on test dataset: ', bal_accuracy)
    print(classification_report(y_test, y_predict))

    overallAUC = roc_auc_score(y_test, y_predict_proba)
    print('AUC:',overallAUC)
    if charts:
        fpr, tpr, thresholds = roc_curve(y_test, y_predict_proba)
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC={auc(fpr, tpr):.5f})')
        plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig('images/ROC_model3.png')
        plt.close()

    fsi = pd.Series(model.feature_importances_, index=X_train.columns)
    fsi_sorted = (fsi/fsi.sum()).sort_values(ascending=False)
    if charts:
        plt.figure(figsize=(15, 5))
        plt.bar(fsi_sorted.index, fsi_sorted.values, color='blue')
        plt.xlabel('Features')
        plt.ylabel('Relative Importance')
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.title("Ranking of feature's importance")
        plt.grid(False)
        plt.tight_layout()
        plt.savefig('images/feature_ranking_model3.png')
        plt.close()
    fsi_df = fsi_sorted.to_frame().reset_index().rename(columns={'index':'Feature',0:'Relative Importance'})
    fsi_df.index += 1
    print(f'\nFeatures ranked by importance for Model 3:\n{fsi_df}')

    if ensemble:
        return fsi, y_predict_proba
    else:
        return