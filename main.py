import re
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from Classification import ClassificationMethods
from FeatureExtraction import WordEmbedding, BasicVec
from visualize import Visualize

class GenderClassification:

    FE_METHOD = "Word2Vec"

    def __init__(self):
        self.data = ""
        self.X_train = ""
        self.X_test = ""
        self.y_train = ""
        self.y_test = ""
        self.t_size = 0.2

    def data_prep(self):
        self.data = pd.read_csv("Tweet_Dataset.csv")

        le = LabelEncoder()
        y = le.fit_transform(self.data.iloc[:,1])

        ps = PorterStemmer()
        #print(set(stopwords.words("turkish")))
        processed_tweet = []
        for i in range(0, 114, 1):
            tweet = re.sub("[^a-zA-Z]", " ", self.data["Tweets"][i]) # noktalama işaretlerini siler.
            tweet = tweet.lower() # Tüm kelimeleri küçük harf yapar.
            tweet =  tweet.split() # Cümleleri liste haline çevirir.
             
            tweet = [ps.stem(word) for word in tweet if not word in set(stopwords.words("turkish"))]
            tweet = " ".join(tweet)

            processed_tweet.append(tweet)

        return processed_tweet, y

    def data_split(self, X, y):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size = self.t_size, random_state=0)

    def visualize(self, roc_arr, names, accs, classifier):
        vis = Visualize(self.FE_METHOD)
        vis.visualize_Accuracy(names, accs)
        vis.visualize_Conf_Mat(classifier, names, self.X_test, self.y_test)
        vis.visualize_ROC(roc_arr, names)
        plt.show()
    
    def run(self):
        if self.FE_METHOD == "Word2Vec" or self.FE_METHOD == "Glove":
            
            processed_tweet, y = self.data_prep()
            
            WE = WordEmbedding(self.data, processed_tweet, self.FE_METHOD)
            X = WE.build_model()

            self.data_split(X, y)

            CM = ClassificationMethods(self.FE_METHOD)
            roc_arr, names, accs, classifier = CM.models(self.X_train, self.X_test, self.y_train, self.y_test)

            self.visualize(roc_arr, names, accs, classifier)
        else:
            processed_tweet, y = self.data_prep()

            BV = BasicVec(self.data, processed_tweet, self.FE_METHOD)
            X = BV.tf_idf_vectorizer(processed_tweet)

            self.data_split(X, y)

            CM = ClassificationMethods(self.FE_METHOD)
            roc_arr, names, accs, classifier = CM.models(self.X_train, self.X_test, self.y_train, self.y_test)

            self.visualize(roc_arr, names, accs, classifier)

if __name__ == "__main__":
    TW = GenderClassification()
    TW.run()
    
    