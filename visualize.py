import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import plot_confusion_matrix

class Visualize:

    def __init__(self, method):
        self.method = method

    def visualize_ROC(self, roc_arr, names):
            plots = []
            countt=0
            plt.figure("Receiver Operating Characteristic - ROC")
            r = 2 if self.method == "Word2Vec" or self.method == "Glove" else 3
            c = 3 if self.method == "Word2Vec" or self.method == "Glove" else 3
            for i in range(r):
                for y in range(c):
                    if self.method == "Word2Vec":
                        if countt==6:
                            break
                    else:
                        if countt==8:
                            break
                    
                    plots.append(plt.subplot2grid((r, c), (i,y)))              
                    plots[countt].plot(roc_arr[countt][0], roc_arr[countt][1], label = "area = {:.4f}".format(roc_arr[countt][2]))
                    plots[countt].set(xlabel = "Specificity(False Positive Rate)", ylabel = "Sensitivity(True Positive Rate)")
                    plots[countt].set_title(names[countt])
                    countt+=1
            plt.tight_layout()
            #plt.show()

    def visualize_Accuracy(self, names, accs):
        log_cols = ["Classifier Method", "Accuracy", "Log Loss"]
        log = pd.DataFrame(columns = log_cols)

        for i in range(6 if self.method == "Word2Vec" or self.method == "Glove" else 8):
            log_entry = pd.DataFrame([[names[i], accs[i]*100, 11]], columns=log_cols)
            log = log.append(log_entry)

        sns.set_color_codes("muted")
        sns.barplot(x = "Accuracy", y = "Classifier Method", data=log, color = "b")
        plt.xlabel("Accuracy %")
        plt.title(self.method)
        #plt.show()

    def visualize_Conf_Mat(self, classifier, names, x_test, y_test):
        r = 2 if self.method == "Word2Vec" or self.method == "Glove" else 2
        c = 3 if self.method == "Word2Vec" or self.method == "Glove" else 4
        fig, axes = plt.subplots(nrows = r, ncols = c, figsize=(10,10))
        countt=0
        for clf, ax in zip(classifier, axes.flatten()):
            if self.method == "Word2Vec":
                if countt==6:
                    break
            else:
                if countt==8:
                    break

            plot_confusion_matrix(clf, 
                            x_test, 
                            y_test, 
                            ax=ax, 
                            cmap='Blues',
                            colorbar = False)
            ax.title.set_text(names[countt])
            countt +=1
        plt.tight_layout()  
        #plt.show()