"""Save the performances of the model.
Save the dataset from the penultimate layer of the network.
This is the datased used in the project "InferringMusicAndVisualArtStyleEvolution[1]".

References:
[1] https://github.com/erikagardini/InferringMusicAndVisualArtStyleEvolution/upload
[2] Bahuleyan, H.
    Music Genre Classification using Machine Learning Techniques.
    arXiv:1804.01149
[3] https://github.com/HareeshBahuleyan/music-genre-classification
"""

import itertools
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, roc_auc_score, matthews_corrcoef, balanced_accuracy_score
import python.utils as utils

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('../results/conf_matr.png')

    return cm


def one_hot_encoder(true_labels, num_records, num_classes):
    temp = np.array(true_labels[:num_records])
    true_labels = np.zeros((num_records, num_classes))
    true_labels[np.arange(num_records), temp] = 1
    return true_labels


def top_n_accuracy(preds, truths):
    giusti_1 = 0
    giusti_2 = 0
    giusti_3 = 0
    conta = 0

    for i in range(len(preds)):
        index = np.argsort(preds[i])
        if index[-1] == truths[i]:
            giusti_1 = giusti_1 + 1
        if index[-1] == truths[i] or index[-2] == truths[i]:
            giusti_2 = giusti_2 + 1
        if index[-1] == truths[i] or index[-2] == truths[i] or index[-3] == truths[i]:
            giusti_3 = giusti_3 + 1

        conta = conta + 1

    accuracy_top_1 = giusti_1 / conta
    accuracy_top_2 = giusti_2 / conta
    accuracy_top_3 = giusti_3 / conta

    print("TOP1: ")
    print(accuracy_top_1)
    print("TOP2: ")
    print(accuracy_top_2)
    print("TOP3: ")
    print(accuracy_top_3)

    return accuracy_top_1, accuracy_top_2, accuracy_top_3


label_dict = {'classical': 0,
              'baroque': 1,
              'opera': 2,
              'medieval': 3,
              'jazz': 4,
              'rock': 5
              }

[train_files, train_labels, test_files, test_labels, val_files, val_labels] = utils.load_input()
pred = np.load("../results/prediction.npy")
pred_probs = np.load("../results/prediction_probabilities.npy")

[top1, top2, top3] = top_n_accuracy(pred_probs, test_labels)

conf_matrx = plot_confusion_matrix(confusion_matrix(y_true=test_labels[:len(pred)], y_pred=pred), classes=label_dict.keys())

file = open("../results/res.txt","w")
for el in conf_matrx:
    file.write(str(el))
    file.write("\n")

file.write("\n\n\n")

file.write("TOP_1: ")
file.write(str(top1))
file.write("\n")
file.write("TOP_2: ")
file.write(str(top2))
file.write("\n")
file.write("TOP_3: ")
file.write(str(top3))
file.write("\n")
file.write('Test Set Accuracy =  {0:.2f}\n'.format(accuracy_score(y_true=test_labels[:len(pred)], y_pred=pred)))
file.write('Test Set Matthews Coef =  {0:.2f}\n'.format(matthews_corrcoef(y_true=test_labels[:len(pred)], y_pred=pred)))
file.write('Test Set F-score =  {0:.2f}\n'.format(f1_score(y_true=test_labels[:len(pred)], y_pred=pred, average='macro')))

one_hot_true = one_hot_encoder(test_labels, len(pred), len(label_dict))
file.write('ROC AUC = {0:.3f}\n'.format(roc_auc_score(y_true=one_hot_true, y_score=pred_probs, average='macro')))
file.close()


