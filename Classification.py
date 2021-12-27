from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score

class ClassificationMethods:

    def __init__(self, method):
        self.method = method

    def models(self, x_train, x_test, y_train, y_test):

        if self.method == "Word2Vec" or self.method == "Glove":
            classifier = [LogisticRegression(),
                    KNeighborsClassifier(n_neighbors = 5, metric = "minkowski"),
                    SVC(kernel = "rbf", probability = True),
                    XGBClassifier(use_label_encoder = False),
                    DecisionTreeClassifier(criterion = "gini"),
                    RandomForestClassifier(n_estimators = 100, criterion = "gini")]
        else:
            classifier = [XGBClassifier(use_label_encoder = False),
                        KNeighborsClassifier(n_neighbors = 5, metric = "minkowski"),
                        SVC(kernel = "rbf", probability = True),
                        LogisticRegression(),
                        MultinomialNB(),
                        MLPClassifier(solver = "sgd"),
                        DecisionTreeClassifier(criterion = "gini"),
                        RandomForestClassifier(n_estimators = 100, criterion = "gini")]

        accs = []
        names = []
        roc_arr = []
        for clf in classifier:
            name = clf.__class__.__name__
            if name == "XGBClassifier":
                clf.fit(x_train, y_train, eval_metric='map')
            else:
                clf.fit(x_train, y_train)

            print("="*30)
            print(name)
            print("****Results****")

            predictions = clf.predict(x_test)
            acc = accuracy_score(y_test, predictions)

            proba = clf.predict_proba(x_test)
            fpr, tpr, _ = roc_curve(y_test, proba[:,1], pos_label = 1)
            auc = roc_auc_score(y_test, proba[:,1], multi_class = "ovr")

            print("Accuracy: {:.4%}".format(acc))
            print("AUC: {:.4%}".format(auc))

            arr=[]
            arr.append(fpr)
            arr.append(tpr)
            arr.append(auc)
            roc_arr.append(arr)

            names.append(name)
            accs.append(acc)

        return roc_arr, names, accs, classifier

if __name__ == "__main__":
    clf = ClassificationMethods()