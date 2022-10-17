from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys

def load_data(dir, type):
    data_file = open(f'{dir}/{type}.pickle', 'rb')
    data = pickle.load(data_file)
    data_file.close()

    s = data['data'].shape
    data['data'] = data['data'].reshape(s[0], np.product(s[1:]))

    data['data'] = data['data'].astype(np.double)
    data['labels'] = data['labels'].astype(np.double).flatten()

    data['data'] = np.interp(data['data'], (0,255), (0,1))
    
    return data

def cross_validate(model, data, labels, degree):
    scores = []
    kf = KFold(n_splits=degree, shuffle=True)
    for train_idx, test_idx in kf.split(data):
        model.fit(data[train_idx], labels[train_idx])
        preds = model.predict(data[test_idx])
        score = accuracy_score(labels[test_idx], preds)
        scores.append(score)

    return scores

if __name__ == "__main__":
    train_data = load_data(sys.argv[1], 'train_data')
    test_data = load_data(sys.argv[2], 'test_data')

    C = [1e-5,1e-3,1,5,10]
    cv_scores = []
    for c in C:
        clf = svm.SVC(gamma=0.001, C=c)
        scores = cross_validate(clf, train_data['data'], train_data['labels'], 5)
        print(scores)
        cv_scores.append(scores)

    ts_scores = []
    for c in C:
        clf = svm.SVC(gamma=0.001, C=c)
        clf.fit(train_data['data'], train_data['labels'])
        ts_scores.append(accuracy_score(test_data['labels'], clf.predict(test_data['data'])))

    cv_scores = np.array(cv_scores)
    ts_scores = np.array(ts_scores)
    means = cv_scores.mean(axis=1)
    stdevs = cv_scores.std(axis=1)

    plt.plot(C, ts_scores, label='test')
    plt.errorbar(C, means, yerr=stdevs, capsize=3, label='CV')
    plt.xscale('log')
    plt.xlabel('C')
    plt.ylabel('Accuracy')
    plt.title('Accuracy v/s C')
    plt.legend()
    plt.savefig('acc.pdf', bbox_inches='tight')
