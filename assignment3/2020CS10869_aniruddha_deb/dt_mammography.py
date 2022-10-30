import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
import pandas as pd
import matplotlib as mpl 
import matplotlib.pyplot as plt

import sys
from contextlib import redirect_stdout

def preprocess(dataset):
    df = dataset.replace({'?': None}).dropna()
    return df[['Age', 'Shape', 'Margin', 'Density']], df['Severity']

def no_impute(dataset):
    df = dataset.replace({'?': np.nan}).apply(pd.to_numeric)
    return df[['Age', 'Shape', 'Margin', 'Density']], df['Severity']

def preprocess_impute(dataset):
    df = dataset.replace({'?': np.nan}).apply(pd.to_numeric)
    df = df.fillna(df.mode().transpose()[0])
    return df[['Age', 'Shape', 'Margin', 'Density']], df['Severity']

def oob_score(estimator, X_test, y_test):
    return estimator.oob_score_

if __name__ == "__main__":

    X_train, y_train = preprocess(pd.read_csv(sys.argv[1]))
    X_val, y_val = preprocess(pd.read_csv(sys.argv[2]))
    X_test, y_test = preprocess(pd.read_csv(sys.argv[3]))
    outpath = sys.argv[3]
    part = sys.argv[4]
    mpl.rcParams["savefig.directory"] = outpath

    with open(f"{outpath}/1_{part}.txt", "w") as logfile:
        with redirect_stdout(logfile):
            if part == 'a':
                clf = tree.DecisionTreeClassifier()
                clf = clf.fit(X_train, y_train)

                fig, ax = plt.subplots(figsize=(20,10), dpi=150)
                ret = tree.plot_tree(clf, ax=ax, feature_names=['Age', 'Shape', 'Margin', 'Density'], fontsize=14, filled=True, label='root', rounded=True, max_depth=3)
                plt.savefig('tree.pdf', bbox_inches='tight')

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

                fig, ax = plt.subplots(figsize=(20,10), dpi=150)
                ret = tree.plot_tree(grid_searcher.best_estimator_, ax=ax, feature_names=['Age', 'Shape', 'Margin', 'Density'], fontsize=14, filled=True, label='root', rounded=True) 
                plt.savefig('tree_optimal.pdf', bbox_inches='tight')
            elif part == 'c':
                clf = tree.DecisionTreeClassifier()
                path = clf.cost_complexity_pruning_path(X_train, y_train)
                ccp_alphas, impurities = path.ccp_alphas, path.impurities

                fig, ax = plt.subplots(figsize=(8,6), dpi=150)
                ax.plot(ccp_alphas[:-1], impurities[:-1], marker="o", drawstyle="steps-post")
                ax.set_xlabel("effective alpha")
                ax.set_ylabel("total impurity of leaves")
                ax.set_title("Total Impurity vs effective alpha for training set")
                plt.savefig('impurity_vs_alpha.pdf', bbox_inches='tight')

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
                fig, ax = plt.subplots(2, 1, figsize=(6,6), dpi=150)
                ax[0].plot(ccp_alphas, node_counts, marker=".")
                ax[0].set_xlabel("alpha")
                ax[0].set_ylabel("number of nodes")
                ax[0].set_title("Number of nodes vs alpha")
                ax[1].plot(ccp_alphas, depth, marker=".")
                ax[1].set_xlabel("alpha")
                ax[1].set_ylabel("depth of tree")
                ax[1].set_title("Depth vs alpha")
                fig.tight_layout()
                plt.savefig('nodes_vs_alpha.pdf', bbox_inches='tight')

                train_scores = [clf.score(X_train, y_train) for clf in clfs]
                val_scores = [clf.score(X_val, y_val) for clf in clfs]
                test_scores = [clf.score(X_test, y_test) for clf in clfs]

                fig, ax = plt.subplots(figsize=(6,4), dpi=150)
                ax.set_xlabel("alpha")
                ax.set_ylabel("accuracy")
                ax.set_title("Accuracy vs alpha for training and testing sets")
                ax.plot(ccp_alphas, train_scores, marker=".", label="train")
                ax.plot(ccp_alphas, val_scores, marker=".", label="val")
                ax.plot(ccp_alphas, test_scores, marker=".", label="test")
                ax.legend()
                plt.savefig('accuracy_vs_alpha.pdf', bbox_inches='tight')

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

                fig, ax = plt.subplots(figsize=(15,7), dpi=150)
                ret = tree.plot_tree(best_tree, ax=ax, feature_names=['Age', 'Shape', 'Margin', 'Density'], fontsize=14, filled=True, label='root', rounded=True, max_depth=3) 
                plt.savefig('tree_best_pruned.pdf', bbox_inches='tight')

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

                X_train, y_train = preprocess_impute(pd.read_csv(sys.argv[1]))
                X_val, y_val = preprocess_impute(pd.read_csv(sys.argv[2]))
                X_test, y_test = preprocess_impute(pd.read_csv(sys.argv[3]))

                clf = tree.DecisionTreeClassifier()
                clf = clf.fit(X_train, y_train)

                fig, ax = plt.subplots(figsize=(20,10), dpi=150)
                ret = tree.plot_tree(clf, ax=ax, feature_names=['Age', 'Shape', 'Margin', 'Density'], fontsize=14, filled=True, label='root', rounded=True, max_depth=3)
                plt.savefig('tree.pdf', bbox_inches='tight')

                train_acc = clf.score(X_train, y_train)
                val_acc   = clf.score(X_val, y_val)
                test_acc  = clf.score(X_test, y_test)
                                           
                print("Part (a):")
                print(f"Train accuracy: {train_acc:.4f}")
                print(f"Validation accuracy: {val_acc:.4f}")
                print(f"Test accuracy: {test_acc:.4f}")

                grid_searcher = GridSearchCV(clf, {'max_depth': range(4,11,2), 'min_samples_split': range(2,6), 'min_samples_leaf': range(1,6)})
                grid_searcher.fit(X_train, y_train)
                print(grid_searcher.best_params_)

                train_acc = accuracy_score(y_train, grid_searcher.predict(X_train))
                val_acc   = accuracy_score(y_val,   grid_searcher.predict(X_val))
                test_acc  = accuracy_score(y_test,  grid_searcher.predict(X_test))
                                           
                print("Part (b):")
                print(f"Train accuracy: {train_acc:.4f}")
                print(f"Validation accuracy: {val_acc:.4f}")
                print(f"Test accuracy: {test_acc:.4f}")

                clf = tree.DecisionTreeClassifier()
                path = clf.cost_complexity_pruning_path(X_train, y_train)
                ccp_alphas, impurities = path.ccp_alphas, path.impurities

                print("Part (c):")
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

                clf = RandomForestClassifier(bootstrap=True, oob_score=True)
                clf.fit(X_train, y_train)
                clf.score(X_val, y_val)

                gs = GridSearchCV(clf, {'n_estimators': range(50,201,50), 'max_features': range(1,5), 'min_samples_split': range(2,6)}, scoring=oob_score)
                gs.fit(X_train, y_train)

                clf_best = gs.best_estimator_
                print("Part (d):")
                print(f"Train accuracy: {accuracy_score(y_train, clf_best.predict(X_train)):.4f}")
                print(f"Out of bag accuracy: {clf_best.oob_score_:.4f}")
                print(f"Validation accuracy: {accuracy_score(y_val, clf_best.predict(X_val)):.4f}")
                print(f"Test accuracy: {accuracy_score(y_test, clf_best.predict(X_test)):.4f}")

                print(gs.best_params_)

            elif part == 'f':

                X_train_raw, y_train_raw = no_impute(pd.read_csv(sys.argv[1]))
                X_val_raw, y_val_raw = no_impute(pd.read_csv(sys.argv[2]))
                X_test_raw, y_test_raw = no_impute(pd.read_csv(sys.argv[3]))

                clf = XGBClassifier()
                gs = GridSearchCV(clf, {'n_estimators': range(10,51,10), 'subsample': np.arange(0.1,0.61,0.1), 'max_depth': range(4,11)})
                gs.fit(X_train_raw, y_train_raw)
                print(gs.best_params_)
                print(f"Train accuracy: {gs.score(X_train_raw, y_train_raw):.4f}")
                print(f"Validation accuracy: {gs.score(X_val_raw, y_val_raw):.4f}")
                print(f"Test accuracy: {gs.score(X_test_raw, y_test_raw):.4f}")
