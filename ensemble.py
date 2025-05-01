import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, roc_auc_score, classification_report,accuracy_score, balanced_accuracy_score, make_scorer

def balanced_logloss(y_test, y_predict_proba):
    
    y_test = y_test.to_numpy()
    N0 = np.sum(1 - y_test)
    N1 = np.sum(y_test)
    
    p1 = np.clip(y_predict_proba, 1e-15, 1 - 1e-15)
    p0 = 1 - p1
    
    log_loss_0 = -np.sum((1 - y_test) * np.log(p0)) / N0
    log_loss_1 = -np.sum(y_test * np.log(p1)) / N1
    
    return (log_loss_0 + log_loss_1)/2

def aggregate_prediction(y_proba, agg_ft_importance, y_test, charts=False):
    mean_proba = np.mean(y_proba, axis=1)
    agg_pred = (mean_proba>0.5).astype(int)
    confusion = metrics.confusion_matrix(y_test, agg_pred)
    print('\nConfusion matrix after aggregation:\n', confusion)
    if charts:
        confusion_df = pd.DataFrame(confusion, columns=np.unique(y_test), index = np.unique(y_test))
        plt.figure(figsize = (3,3))
        plt.rcParams.update({'font.size': 15})
        sns.heatmap(confusion_df, cmap = 'Blues', annot=True, fmt='g', square=True, linewidths=.5, cbar=False)
        plt.xlabel('Predicted')
        plt.ylabel('Expected')
        plt.tight_layout()
        plt.savefig('images/confusion_matrix_aggregated.png')
        plt.close()
    print('\nBalanced Log Loss after aggregation:{}'.format(balanced_logloss(y_test, mean_proba)))
    bal_accuracy = balanced_accuracy_score(y_test,agg_pred)
    print('\nBalanced_Accuracy_score after aggregation: ', bal_accuracy)
    print(classification_report(y_test, agg_pred))

    overallAUC = roc_auc_score(y_test, mean_proba)
    print('AUC:',overallAUC)
    if charts:
        fpr, tpr, thresholds = roc_curve(y_test, mean_proba)
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC={auc(fpr, tpr):.5f})')
        plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig('images/ROC_aggregated.png')
        plt.close()

    fsi = pd.DataFrame(agg_ft_importance).T
    fsi = fsi.mean(axis=1)
    fsi = (fsi/fsi.sum()).sort_values(ascending=False)
    if charts:
        plt.figure(figsize=(15, 5))
        plt.bar(fsi.index, fsi.values, color='blue')
        plt.xlabel('Features')
        plt.ylabel('Relative Importance')
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.title("Ranking of feature's importance")
        plt.grid(False)
        plt.tight_layout()
        plt.savefig('images/feature_ranking_aggregated.png')
        plt.close()
    fsi = fsi.to_frame().reset_index().rename(columns={'index':'Feature',0:'Relative Importance'})
    fsi.index += 1
    print(f'\nFeatures ranked by importance after aggregation:\n{fsi}')

    return
    


