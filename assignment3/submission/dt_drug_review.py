import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
import pandas as pd

import sys
from contextlib import redirect_stdout
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import scipy.sparse as sps
import re

cv_review = CountVectorizer(stop_words='english', strip_accents='ascii')
cv_condition = CountVectorizer(stop_words='english', strip_accents='ascii')

def preprocess_train(dataset):
    dataset['review'] = dataset['review'].replace({np.nan: ''})
    dataset['condition'] = dataset['condition'].replace({np.nan: ''})
    review_enc = cv_review.fit_transform(dataset['review'])       # returns a sparse matrix
    condition_enc = cv_condition.fit_transform(dataset['condition']) # returns a sparse matrix

    dates = pd.to_datetime(dataset['date'])
    date_mat = sps.csr_matrix(np.hstack([dates.dt.year.to_numpy().reshape(-1,1), dates.dt.month.to_numpy().reshape(-1,1), dates.dt.day.to_numpy().reshape(-1,1)]))
    usefulcnt_mat = sps.csr_matrix(dataset['usefulCount'].to_numpy().reshape(-1,1))
    return sps.hstack([condition_enc, review_enc, date_mat, usefulcnt_mat]), dataset['rating'].to_numpy()

def preprocess_test(dataset):
    dataset['review'] = dataset['review'].replace({np.nan: ''})
    dataset['condition'] = dataset['condition'].replace({np.nan: ''})
    review_enc = cv_review.transform(dataset['review'])       # returns a sparse matrix
    condition_enc = cv_condition.transform(dataset['condition']) # returns a sparse matrix

    dates = pd.to_datetime(dataset['date'])
    date_mat = sps.csr_matrix(np.hstack([dates.dt.year.to_numpy().reshape(-1,1), dates.dt.month.to_numpy().reshape(-1,1), dates.dt.day.to_numpy().reshape(-1,1)]))
    usefulcnt_mat = sps.csr_matrix(dataset['usefulCount'].to_numpy().reshape(-1,1))
    return sps.hstack([condition_enc, review_enc, date_mat, usefulcnt_mat]), dataset['rating'].to_numpy()

def oob_score(estimator, X_test, y_test):
    return estimator.oob_score_

if __name__ == "__main__":

    X_train, y_train = preprocess_train(pd.read_csv(sys.argv[1]))
    X_val, y_val = preprocess_test(pd.read_csv(sys.argv[2]))
    X_test, y_test = preprocess_test(pd.read_csv(sys.argv[3]))
    outpath = sys.argv[3]
    part = sys.argv[4]

    with open(f"{outpath}/2_{part}.txt", "w") as logfile:
        with redirect_stdout(logfile):
            if part == 'a':
                clf = tree.DecisionTreeClassifier()
                clf = clf.fit(X_train, y_train)

                train_acc = clf.score(X_train, y_train)
                val_acc   = clf.score(X_val, y_val)
                test_acc  = clf.score(X_test, y_test)
                                           
                print(f"Train accuracy: {train_acc:.4f}")
                print(f"Validation accuracy: {val_acc:.4f}")
                print(f"Test accuracy: {test_acc:.4f}")
            elif part == 'b':
                clf = tree.DecisionTreeClassifier()
                grid_searcher = GridSearchCV(clf, {'max_depth': range(4,11,2), 'min_samples_split': range(2,6), 'min_samples_leaf': range(1,6)})
                grid_searcher.fit(X_train, y_train)
                print(grid_searcher.best_params_)

                train_acc = accuracy_score(y_train, grid_searcher.predict(X_train))
                val_acc   = accuracy_score(y_val,   grid_searcher.predict(X_val))
                test_acc  = accuracy_score(y_test,  grid_searcher.predict(X_test))
                                           
                print(f"Train accuracy: {train_acc:.4f}")
                print(f"Validation accuracy: {val_acc:.4f}")
                print(f"Test accuracy: {test_acc:.4f}")

            elif part == 'c':
                clf = tree.DecisionTreeClassifier()
                path = clf.cost_complexity_pruning_path(X_train, y_train)
                ccp_alphas, impurities = path.ccp_alphas, path.impurities

                clfs = []
                for ccp_alpha in ccp_alphas:
                    clf = tree.DecisionTreeClassifier(ccp_alpha=ccp_alpha)
                    clf.fit(X_train, y_train)
                    clfs.append(clf)
                print(
                    "Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
                        clfs[-1].tree_.node_count, ccp_alphas[-1]
                    )
                )

                clfs = clfs[:-1]
                ccp_alphas = ccp_alphas[:-1]

                node_counts = [clf.tree_.node_count for clf in clfs]
                depth = [clf.tree_.max_depth for clf in clfs]

                train_scores = [clf.score(X_train, y_train) for clf in clfs]
                val_scores = [clf.score(X_val, y_val) for clf in clfs]
                test_scores = [clf.score(X_test, y_test) for clf in clfs]

                best_tree_idx = np.argmax(val_scores)
                print(
                    "Number of nodes in the best tree is: {} with ccp_alpha: {}".format(
                        clfs[best_tree_idx].tree_.node_count, ccp_alphas[best_tree_idx]
                    )
                )

                best_tree = clfs[best_tree_idx]
                best_tree_alpha = ccp_alphas[best_tree_idx]

                train_acc = best_tree.score(X_train, y_train)
                val_acc = best_tree.score(X_val, y_val)
                test_acc = best_tree.score(X_test, y_test)

                print(f"Train accuracy on best clf: {train_acc:.4f}")
                print(f"Validation accuracy on best clf: {val_acc:.4f}")
                print(f"Test accuracy on best clf: {test_acc:.4f}")

            elif part == 'd':

                clf = RandomForestClassifier(bootstrap=True, oob_score=True)
                clf.fit(X_train, y_train)
                clf.score(X_val, y_val)

                gs = GridSearchCV(clf, {'n_estimators': range(50,201,50), 'max_features': range(1,5), 'min_samples_split': range(2,6)}, scoring=oob_score)
                gs.fit(X_train, y_train)

                clf_best = gs.best_estimator_
                print(f"Train accuracy: {accuracy_score(y_train, clf_best.predict(X_train)):.4f}")
                print(f"Out of bag accuracy: {clf_best.oob_score_:.4f}")
                print(f"Validation accuracy: {accuracy_score(y_val, clf_best.predict(X_val)):.4f}")
                print(f"Test accuracy: {accuracy_score(y_test, clf_best.predict(X_test)):.4f}")

                print(gs.best_params_)

            elif part == 'e':

                clf = XGBClassifier()
                gs = GridSearchCV(clf, {'n_estimators': range(10,51,10), 'subsample': np.arange(0.1,0.61,0.1), 'max_depth': range(4,11)})
                gs.fit(X_train, y_train)
                print(gs.best_params_)
                print(f"Train accuracy: {gs.score(X_train, y_train):.4f}")
                print(f"Validation accuracy: {gs.score(X_val, y_val):.4f}")
                print(f"Test accuracy: {gs.score(X_test, y_test):.4f}")

            elif part == 'f':
                print("Not implemented")

            elif part == 'g':
                print("Not implemented")
