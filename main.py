"""main.py

Code scaffolding

"""

import os
import string
from math import *
import nltk
from nltk.corpus import brown
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize.treebank import TreebankWordDetokenizer

"""
Written by Vitor Mouzinho
date: October 9th ,2019
class: 193LING-131A-1 : Introduction to Natural Language Processing with Python
PA02
"""


# reads file and returns text obj if file is dir then returns all file within dir as one text obj
def read_text(path):
    if os.path.isfile(path):
        file = open(path, "r")
        corpa = file.read()
        nltk_text = nltk.Text(nltk.word_tokenize(corpa));
        file.close()
        return nltk_text
    elif os.path.isdir(path):
        corpa = ""
        for wsj in os.listdir(path):
            file = open("data/wsj/" + wsj, "r")
            corpa += file.read() + "\n"
            file.close()
        nltk_text = nltk.Text(nltk.word_tokenize(corpa))
        return nltk_text
    else:
        return None


# finds how many tokens are available
def token_count(text):
    if text is None:
        return None
    return len(text)


# uses set to find type count
def type_count(text):
    if text is None:
        return None
    return len(set(text))


# uses built in function of text get count of sentence
def sentence_count(text):
    if text is None:
        return None
    count = text.count(".")
    count += text.count("!")
    count += text.count("?")
    return count


# finds the top most common content_words
def most_frequent_content_words(text):
    if text is None:
        return None
    return nltk.FreqDist(normalize_corpus(text)).most_common(25)


# removes stop words and punctuations
def normalize_corpus(text):
    if text is None:
        return None
    normalized_corpus = remove_stop_words(TreebankWordDetokenizer().detokenize(text))
    normalized_corpus = nltk.re.sub(r'[^a-zA-Z0-9.\s]', ' ', normalized_corpus)
    tokens = [token for token in normalized_corpus.split(" ") if token != ""]
    return tokens


# finds most frequent bigrams
def most_frequent_bigrams(text):
    if text is None:
        return None
    tokens = normalize_corpus(text)
    output = list(nltk.ngrams(tokens, 2))
    return nltk.FreqDist(output).most_common(25)


# removes stop words and punctuations
def remove_stop_words(text):
    if text is None:
        return None
    stop_words = set(stopwords.words('english'))
    word_tokens = nltk.word_tokenize(text)
    filtered_sentence = ""
    punctuations = string.punctuation
    for w in word_tokens:
        if w.lower() not in stop_words:
            if w not in punctuations:
                if len(w) > 1:
                    filtered_sentence += w + " "
    return filtered_sentence


class Vocabulary:
    list_of_words = []
    text = ""

    def __init__(self, text):
        self.list_of_words = text.vocab()
        self.text = text

    def frequency(self, word):
        if word in self.list_of_words.keys():
            return self.list_of_words.get(word)
        else:
            return 0

    def pos(self, word):
        if wordnet.synsets(word):
            syn = wordnet.synsets(word)[0]
            return syn.pos()
        else:
            return None;

    def gloss(self, word):
        if wordnet.synsets(word):
            syn = wordnet.synsets(word)[0]
            return syn.definition()
        else:
            return None

    def quick(self, word):
        if self.text.concordance(word):
            return self.text.concordance(word)
        else:
            return None


categories = ('adventure', 'fiction', 'government', 'humor', 'news')


# for each categories finds the similarity
def compare_to_brown(text):
    if text is None:
        return None
    results = ""
    for w in categories:
        results += w + " " + str(set_calculate(text, w)) + "\n"
    return results


# finds the words with the same words in both corpus and then does calculation on its frequencies
def set_calculate(text, catergorie):
    subject = brown.words(categories=catergorie)

    subject_frequency = nltk.FreqDist(subject)
    x = ""

    for m in subject_frequency.items():
        fm = m[0].translate(string.punctuation)
        if (fm.isalpha()):
            for z in range(int(m[1])):
                x += fm.lower() + " "

    x = ' '.join(x.split())

    last = nltk.FreqDist(nltk.word_tokenize(x))
    z = ""

    for m in text.vocab().items():
        fm = m[0].translate(string.punctuation)
        if (fm.isalpha()):
            for l in range(int(m[1])):
                z += fm.lower() + " "

    z = ' '.join(z.split())
    mz = nltk.FreqDist(nltk.word_tokenize(z))

    l = []

    for w in last.keys():
        if (mz.keys().__contains__(w)):
            l.append((last.get(w), mz.get(w)))
    l2 = []
    l3 = []

    for w in l:
        l2.append(w[0])
        l3.append(w[1])
    return cosine_similarity(l3, l2)


# used to calculate square root of algorithm
def square_rooted(x):
    return round(sqrt(sum([a * a for a in x])), 3)


# used to calculate cosine_similarity
def cosine_similarity(x, y):
    numerator = sum(a * b for a, b in zip(x, y))
    denominator = square_rooted(x) * square_rooted(y)
    return round(numerator / float(denominator), 2)


if __name__ == '__main__':
    text = read_text('data/emma.txt')
    content_words = compare_to_brown(text)
    print(content_words)
