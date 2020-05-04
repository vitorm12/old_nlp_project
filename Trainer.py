from random import shuffle

import nltk
from nltk.corpus import movie_reviews, stopwords
import time

class Trainer:

    def __init__(self):
        self.binaryTrainer = None
        self.raw_countTrainer = None

    def create_word_features(self, words):
        stp = stopwords.words("english");
        useful_words = [word for word in words if word not in stp]
        print(useful_words)
        my_dict = dict([(word, True) for word in useful_words])
        return my_dict


    def create_pos_neg(self,prefix, sent):
        reviews = []
        fileids = movie_reviews.fileids(prefix)
        for fileid in fileids:
            words = movie_reviews.words(fileid);
            reviews.append((self.create_word_features(words), sent))
        return reviews

    def create_binary_trainer(self):
        print("Creating binary classifier in classifiers/bayes-all-words.jbl")
        start_time = time.time()
        neg_reviews = self.create_pos_neg('neg', "neg")
        pos_reviews = self.create_pos_neg('pos', "pos");
        train = neg_reviews[:750] + pos_reviews[:750]
        test = neg_reviews[750:] + pos_reviews[750:]
        classifier = nltk.NaiveBayesClassifier.train(train)
        self.binaryTrainer = classifier
        acc = nltk.classify.util.accuracy(classifier, test)
        end = time.time() - start_time
        print(acc, " ", end)

    def raw_count(self):
        pos = self.frequency_distrub_tupple("pos")
        neg = self.frequency_distrub_tupple("neg")
        train = pos[:750] + neg[:750];
        test = pos[750:] + neg[750:];
        c = nltk.NaiveBayesClassifier.train(train);
        acc = nltk.classify.util.accuracy(c, test)
        print(acc)

    def frequency_distrub_tupple(self,suffix):
        documents = movie_reviews.fileids(suffix)
        review_tupples = []
        for fileid in documents:
            words = movie_reviews.words(fileid);
            fr = nltk.FreqDist(words)
            review_tupples.append((fr, suffix))
        return review_tupples

    def create_senti(self):
        documents = movie_reviews.fileids("pos")
        review_tupples = []
        for fileid in documents:
            pos = nltk.pos_tag(movie_reviews.words(fileid))
            print(pos)


