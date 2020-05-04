from pickle import dump, load

import nltk
from nltk.corpus import movie_reviews, stopwords
import time
from nltk.corpus import sentiwordnet as swn
import pickle





class Bayes:

    def __init__(self,read):
        self.binaryTrainer = None
        self.raw_countTrainer = None
        self.sentiTrainer = None
        self.negation = None
        if read:
            bt = open("classifiers/bayes-Negation.pkl", 'rb')
            self.negation = pickle.load(bt)
            bt.close()

            sen = open("classifiers/bayes-sentiwordnet.pkl", 'rb')
            self.sentiTrainer = pickle.load(sen)
            sen.close()

            bw = open("classifiers/bayes-binary-words.pkl", 'rb')
            self.binaryTrainer = pickle.load(bw)
            bw.close()

            raw = open("classifiers/bayes-raw-count.pkl", 'rb')
            self.raw_countTrainer = pickle.load(raw)
            raw.close()


    def create_word_features(self, words):
        stp = stopwords.words("english");
        useful_words = [word for word in words if word not in stp]
        my_dict = dict([(word, True) for word in useful_words])
        return my_dict

    def create_pos_neg(self, prefix, sent):
        reviews = []
        fileids = movie_reviews.fileids(prefix)
        for fileid in fileids:
            words = movie_reviews.words(fileid);
            reviews.append((self.create_word_features(words), sent))
        return reviews

    def get_pos_neg(self,file):
        f = open(file, "r")
        data = nltk.word_tokenize(f.read())
        classify = self.create_word_features(data)
        f.close()
        print(self.binaryTrainer.classify(classify))

    def get_raw(self,file):
        f = open(file, "r")
        data = nltk.word_tokenize(f.read())
        classify = self.create_word_features(data)
        f.close()
        print(self.raw_countTrainer.classify(classify))

    def get_neg(self,file):
        f = open(file, "r")
        data = nltk.word_tokenize(f.read())
        classify = self.create_word_features(data)
        f.close()
        print(self.negation.classify(classify))

    def get_senti(self,file):
        f = open(file, "r")
        data = nltk.word_tokenize(f.read())
        classify = self.create_word_features(data)
        f.close()
        print(self.sentiTrainer.classify(classify))



    def create_binary_trainer(self):
        print("Creating Bayes classifier in classifiers/bayes-binary-words.pkl")
        start_time = time.time()
        neg_reviews = self.create_pos_neg('neg', "neg")
        pos_reviews = self.create_pos_neg('pos', "pos");
        train = neg_reviews[:750] + pos_reviews[:750]
        test = neg_reviews[750:] + pos_reviews[750:]
        classifier = nltk.NaiveBayesClassifier.train(train)
        self.binaryTrainer = classifier
        acc = nltk.classify.util.accuracy(classifier, test)
        end = time.time() - start_time
        print("  Elapsed time: ",end,"s","\n"," Accuracy: ",acc)

    def raw_count(self):
        print("Creating Bayes classifier in classifiers/bayes-raw-count.pkl")
        start_time = time.time()
        pos = self.frequency_distrub_tupple("pos")
        neg = self.frequency_distrub_tupple("neg")
        train = pos[:750] + neg[:750];
        test = pos[750:] + neg[750:];
        c = nltk.NaiveBayesClassifier.train(train);
        self.raw_countTrainer = c
        acc = nltk.classify.util.accuracy(c, test)
        end = time.time() - start_time
        print("  Elapsed time: ",end,"s","\n"," Accuracy: ",acc)

    def frequency_distrub_tupple(self, suffix):
        documents = movie_reviews.fileids(suffix)
        review_tupples = []
        for fileid in documents:
            words = movie_reviews.words(fileid);
            fr = nltk.FreqDist(words)
            review_tupples.append((fr, suffix))
        return review_tupples

    def create_senti(self):
        print("Creating Bayes classifier in classifiers/bayes-sentiwordnet.pkl")
        start_time = time.time()
        final_pos = self.create_senti_pos_neg("pos")
        final_neg = self.create_senti_pos_neg("neg")
        train = final_pos[:750] + final_neg[:750];
        test = final_pos[750:] + final_neg[750:];
        c = nltk.NaiveBayesClassifier.train(train);
        self.sentiTrainer = c
        acc = nltk.classify.util.accuracy(c, test)
        end = time.time() - start_time
        print("  Elapsed time: ",end,"s","\n"," Accuracy: ",acc)

    def create_senti_pos_neg(self, sent):
        pos = []
        documents = movie_reviews.fileids(sent)
        for fileid in documents:
            tokens = set(movie_reviews.words(fileid))
            movie = list(filter(None, [self.find_senti(a) for a in tokens]))
            pos.append((movie, sent))
        dic = {}
        final_pos = []
        for m in pos:
            for x in m[0]:
                dic[x] = x[1]
            final_pos.append((dic, m[1]))
        return final_pos

    def find_senti(self, a):
        if list(swn.senti_synsets(a)):
            top = list(swn.senti_synsets(a)).pop(0)
            if top.pos_score() > 0.5 or top.neg_score() > 0.5:
                return a, top.obj_score()

    def negation_create(self):
        print("Creating Bayes classifier in classifiers/bayes-Negation.pkl")
        start_time = time.time()
        fixed_pos = self.not_replacement("pos")
        fixed_neg = self.not_replacement("neg")
        n = self.freq_not(fixed_neg,"neg")
        p = self.freq_not(fixed_pos,"pos")
        train = n[:750] + p[:750];
        test = n[750:] + p[750:];
        c = nltk.NaiveBayesClassifier.train(train);
        self.negation = c
        acc = nltk.classify.util.accuracy(c, test)
        end = time.time() - start_time
        print("  Elapsed time: ",end,"s","\n"," Accuracy: ",acc)

    def freq_not(self,list,suffix):
        review_tupples=[]
        for doc in list:
            fr = nltk.FreqDist(doc)
            review_tupples.append((fr, suffix))
        return review_tupples

    def not_replacement(self,sent):
        pos = movie_reviews.fileids(sent)
        fixed_list = []
        for m in pos:
            words = movie_reviews.words(m)
            fixed_word = []
            true = False
            for m in words:
                if true:
                    fixed_word.append("not_" + m)
                    true = False
                else:
                    if m == "not":
                        true = True
                    else:
                        fixed_word.append(m)

            fixed_list.append(fixed_word)
        return fixed_list

    def dump_to_file(self):
        output = open("classifiers/bayes-Negation.pkl", 'wb')
        dump(self.negation,output,-1)
        output.close()

        output = open("classifiers/bayes-sentiwordnet.pkl", 'wb')
        dump(self.sentiTrainer, output, -1)
        output.close()

        output = open("classifiers/bayes-binary-words.pkl", 'wb')
        dump(self.binaryTrainer, output, -1)
        output.close()

        output = open("classifiers/bayes-raw-count.pkl", 'wb')
        dump(self.raw_countTrainer, output, -1)
        output.close()











