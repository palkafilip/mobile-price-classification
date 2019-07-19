import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

def preprocess_data(dataframe, configuration):
	y = np.array(dataframe['price_range'])
	X = np.array(dataframe.drop(['price_range'], axis=1))
	for operation in configuration.operations:
		X = operation.fit_transform(X)
	return train_test_split(X, y, test_size=0.2, random_state=66, stratify=y)

def fit_and_test(clf, clf_name, data, show_graphics=False):
    confusion_matrices = []
    cumsum_accuracy = 0
    cumsum_f1_score = 0
    splits = 5
    kf = StratifiedKFold(n_splits=splits, random_state=1)
    for train_index, test_index in kf.split(data.X,data.Y):
        X_train, X_test = data.X[train_index], data.X[test_index]
        y_train, y_test = data.Y[train_index], data.Y[test_index]

        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        
        cumsum_accuracy += accuracy_score(y_test, y_pred)
        cumsum_f1_score += f1_score(y_test, y_pred, average=None)
        if show_graphics:
            confusion_matrices.append(confusion_matrix(y_test, y_pred))
    
    if show_graphics:
        print(f'Classifier: {clf_name}')
        print(f'Average accuracy: {round((cumsum_accuracy/splits)*100, 2)}%')
        print(f'Average F1 score: {np.round(cumsum_f1_score/splits, 2)}')
        summed_matrix = sum_matrices(confusion_matrices)
        plot_confusion_matrix(summed_matrix, [0,1,2,3], f'{clf_name} confusion matrix')
    
    return cumsum_accuracy/splits

def final_fit_and_test(clf, clf_name, learn_data, test_data):
    clf.fit(learn_data.X, learn_data.Y)
    y_pred = clf.predict(test_data.X)

    accuracy = accuracy_score(test_data.Y, y_pred)
    f1 = f1_score(test_data.Y, y_pred, average=None)
    
    print(f'Classifier: {clf_name}')
    print(f'Accuracy: {round(accuracy * 100, 2)}%')
    print(f'F1 score: {np.round(f1, 2)}')
    plot_confusion_matrix(confusion_matrix(test_data.Y, y_pred), [0,1,2,3], f'{clf_name} confusion matrix')
    
    return accuracy

def sum_matrices(matrices):
    summed_matrix = np.zeros((4,4))
    for matrix in matrices:
        summed_matrix += matrix
    return summed_matrix

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None):
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:,}".format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()