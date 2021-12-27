import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec
from glove import Corpus, Glove

class BasicVec:

    def __init__(self, data, processed_tweet, method):
        self.data = data
        self.method = method
        self.processed_tweet = processed_tweet

    def word_count(self, processed_tweet): # Bag of Words (BOW)
        cv = CountVectorizer(max_features = 50)

        X = cv.fit_transform(processed_tweet).toarray() # count vector

        return X

    def tf_idf_vectorizer(self,processed_tweet):
        tf_idf = TfidfVectorizer()

        X = tf_idf.fit_transform(processed_tweet).toarray()

        return X

class WordEmbedding:

    def __init__(self, data, processed_tweet, method):
        self.data = data
        self.method = method
        self.processed_tweet = processed_tweet

    def Word2Vec(self):
        tokenized_tweet = self.data["Tweets"].apply(lambda x: x.split())

        w2v_model = Word2Vec(tokenized_tweet,
                        min_count=20,
                        window=2,
                        sample=6e-5, 
                        alpha=0.03, 
                        min_alpha=0.0007, 
                        negative=20)

        #w2v_model.build_vocab(tokenized_tweet, progress_per=10000)
        w2v_model.train(tokenized_tweet, total_examples = len(self.processed_tweet), epochs=30) # total_examples=w2v_model.corpus_count

        return w2v_model, tokenized_tweet
    
    def GloveVec(self):
        tokenized_tweet = self.data["Tweets"].apply(lambda x: x.split())

        corpus = Corpus()
        corpus.fit(tokenized_tweet, window = 10)

        glove = Glove(no_components=5, learning_rate=0.05)
        glove.fit(corpus.matrix, epochs = 30, no_threads=4, verbose=True)
        glove.add_dictionary(corpus.dictionary)

        return glove, tokenized_tweet

    def word_vector(self, model, tokens, size):
        vec = np.zeros(size).reshape((1, size))
        count = 0
        for word in tokens:
            try:
                if self.method == "Word2Vec":
                    vec += model.wv[word].reshape(1, size)
                    count += 1
                else:
                    vec += model.word_vectors[model.dictionary[word]].reshape(1, size)
                    count += 1.
            except KeyError:
                continue
        if count != 0:
            vec /= count

        return vec
    
    def get_vector(self, model, tokenized_tweet):
        size = 100 if self.method == "Word2Vec" else 5
        wordvec_arrays = np.zeros((len(tokenized_tweet), size)) 
        for i in range(len(tokenized_tweet)):
            wordvec_arrays[i,:] = self.word_vector(model, tokenized_tweet[i], size)
        wordvec_df = pd.DataFrame(wordvec_arrays)
        wordvec_df.shape

        return wordvec_df
    
    def build_model(self):
        if self.method == "Word2Vec":
            w2v_model, tokenized_tweet = self.Word2Vec()
            model_df = self.get_vector(w2v_model, tokenized_tweet)
        else:
            glove, tokenized_tweet = self.GloveVec()
            model_df = self.get_vector(glove, tokenized_tweet)

        return model_df
