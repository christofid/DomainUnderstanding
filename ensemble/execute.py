import pickle
import os
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from woc import BinaryWoc

from configure import parse_args


def execute_case(data_path, predictions_path, output_path):

    with open(data_path,'rb') as f:
        dataset = pickle.load(f)

    words = dataset['words']

    files = os.listdir(predictions_path)

    groundtruth = None

    predictions = []

    for file in files:
        file_path = os.path.join(predictions_path,file)

        with open(file_path,'rb') as f:
            data = pickle.load(f)

            if groundtruth is None:

                groundtruth=data['labels']

            else:
                for x,y in zip(groundtruth,data['labels']):
                    x.sort()
                    y.sort()
                    if x!=y or len(x)==0:
                        print(x,y)
                assert groundtruth==data['labels']

            pred = [x for x in data['predictions']]
            pred = [1 if y in x else 0 for x in pred for y in words]
            predictions.append(pred)


    predictions = np.array(predictions)

    predictions[predictions==0] = -1

    clw = BinaryWoc()
    scores = clw.get_inference(predictions)

    scores = np.reshape(scores,(len(groundtruth),-1))

    predictions = [ [words[i] for i,x in enumerate(case) if x==1] for case in scores]

    if output_path is not None:
        output = {}
        output['labels'] = groundtruth
        output['predictions'] = predictions

        with open(output_path, 'wb') as f:
            pickle.dump(output, f)

    acc = 0
    for labels,pred in zip(groundtruth, predictions):

        if all([x in labels for x in pred]) and len(pred) == len(labels):
            acc += 1

    acc =  acc / len(predictions)

    labels = [1 if y in x else 0 for x in groundtruth for y in words]
    labels = np.array(labels)
    labels = labels.reshape([-1])

    predictions_bin = ([1 if y in pred else 0 for pred in predictions for y in words])
    predictions_bin = np.array(predictions_bin)

    acc_el = accuracy_score(labels, predictions_bin)
    pre = precision_score(labels, predictions_bin)
    rec = recall_score(labels, predictions_bin)
    f1 = f1_score(labels, predictions_bin)

    print()
    print('Accuracy per instance: {:.4f}'.format(acc))

    print('Accuracy per element: {:.4f}'.format(acc_el))
    print('Precision per element: {:.4f}'.format(pre))
    print('Recall per element: {:.4f}'.format(rec))
    print('F1-score per element: {:.4f}'.format(f1))
    print()


if __name__ == '__main__':

    args = parse_args()
    execute_case(**args)

