import itertools
import operator
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report


def most_common(L):
    # get an iterable of (item, iterable) pairs
    SL = sorted((x, i) for i, x in enumerate(L))
    # print 'SL:', SL
    groups = itertools.groupby(SL, key=operator.itemgetter(0))

    # auxiliary function to get "quality" for an item
    def _auxfun(g):
        item, iterable = g
        count = 0
        min_index = len(L)
        for _, where in iterable:
            count += 1
            min_index = min(min_index, where)
        # print 'item %r, count %r, minind %r' % (item, count, min_index)
        return count, -min_index

    # pick the highest-count/earliest item
    return max(groups, key=_auxfun)[0]


Labels = ['Covid', 'Pneumonia', 'Healthy']


def displayer(patient_pred, patient_test, frame_pred, frame_test, model_name):
    # Show Frame level F1 Score
    print(model_name + ' Frame level F1 Score')
    print(classification_report(frame_pred, frame_test, digits=3))
    # Show Frame level CF
    array = confusion_matrix(frame_pred, frame_test)
    print(array)
    df_cm = pd.DataFrame(array, index=[i for i in Labels],
                         columns=[i for i in Labels])
    ax = plt.axes()
    sn.heatmap(df_cm, annot=True, cmap="YlGnBu")

    ax.set_title(model_name + ' frame level')
    plt.show()
    
    

    # show patient level F1 score
    print(model_name + ' Patient level F1 Score')
    print(classification_report(patient_pred, patient_test, digits=3))
    # Show patient level CF
    array = confusion_matrix(patient_pred, patient_test)
    df_cm = pd.DataFrame(array, index=[i for i in Labels],
                         columns=[i for i in Labels])
    ax = plt.axes()
    sn.heatmap(df_cm, annot=True, cmap="YlGnBu")

    ax.set_title(model_name + ' Patient level')
    plt.show()
    # show patient level CF
