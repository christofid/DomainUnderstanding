from translate import *

import numpy as np
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_case(test_txt, test_labels, words, txt_lang, labels_lang, model, device, max_label_len, output_msg='', save_pred=None):

    acc_ord = 0
    acc_un = 0

    predictions = []
    predictions_bin = []
    confidence = []

    for sentence, case_labels in zip(test_txt, test_labels):
        pred, _, conf = translate_sentence(sentence, txt_lang, labels_lang, model, device, max_len=max_label_len)

        if pred[:-1] == case_labels:
            acc_ord += 1

        if all([x in case_labels for x in pred[:-1]]) and len(pred[:-1]) == len(case_labels):
            acc_un += 1

        predictions.append(pred)
        predictions_bin.append([1 if y in pred else 0 for y in words])
        confidence.append(conf)

    if save_pred is not None:
        output = {}
        output['labels']=test_labels
        output['predictions']=predictions
        output['confidence']= confidence

        with open(save_pred,'wb') as f:
            pickle.dump(output, f)

    labels = [1 if y in x else 0 for x in test_labels for y in words]

    labels = np.array(labels)
    predictions_bin = np.array(predictions_bin)

    acc_ord = acc_ord / len(test_txt)
    acc_un = acc_un / len(test_txt)
    acc_el = accuracy_score(labels.reshape([-1]), predictions_bin.reshape([-1]))
    pre = precision_score(labels.reshape([-1]), predictions_bin.reshape([-1]))
    rec = recall_score(labels.reshape([-1]), predictions_bin.reshape([-1]))
    f1 = f1_score(labels.reshape([-1]), predictions_bin.reshape([-1]))

    print()
    print('{}Accuracy ordered: {:.4f}'.format(output_msg, acc_ord))
    print('{}Accuracy unordered: {:.4f}'.format(output_msg, acc_un))

    print('{}Accuracy per element: {:.4f}'.format(output_msg, acc_el))
    print('{}Precision per element: {:.4f}'.format(output_msg, pre))
    print('{}Recall per element: {:.4f}'.format(output_msg, rec))
    print('{}F1-score per element: {:.4f}'.format(output_msg, f1))
    print()

    return acc_ord, acc_un, acc_el, pre, rec, f1
