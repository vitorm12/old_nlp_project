import nltk
from nltk import collections
from nltk.corpus import brown


COMPILED_BROWN = 'brown.pickle'

BROWN_CATEGORIES = ('adventure', 'fiction', 'government', 'humor', 'news')
text = ""
words = ""


class BrownCorpus(object):

    def __init__(self):
        self.words = nltk.corpus.brown.tagged_words()

        self.text = " ".join(brown.words()).lower()


def nouns_more_common_in_plural_form(bc):
    sentence = ""
    set_singal = set()
    set_plural = set()
    # puts singular and plural nouns into different sets
    for word in bc.words:
        if word[1] == "NN":
            set_singal.add(word[0])
            sentence += " " + word[0]
        elif word[1] == "NNS" and word[0].isalpha():
            set_plural.add(word[0][:-1])
            sentence += " " + word[0]
    sentence = sentence.lower()
    final_set = set()

    # puts all the nouns that are plural and singular into fd
    fd = nltk.FreqDist(nltk.word_tokenize(sentence))
    # adds to final set if the singular form is greater within the plural version
    for m in set_singal:
        if fd.__contains__(m + "s") and fd.get(m + "s") > fd.get(m) and set_plural.__contains__(m):
            final_set.add(m)

    # removes from singal set
    for m in set_singal:
        if set_plural.__contains__(m):
            set_plural.remove(m)
    # adds to final set the remainder
    for m in set_plural:
        final_set.add(m)

    return final_set


def which_word_has_greatest_number_of_distinct_tags(bc):
    word_tag = nltk.defaultdict(set)
    for w, t in bc.words:
        if w.isalpha():
            word_tag[w.lower()].add(t)
    m = max(len(word_tag[w]) for w in word_tag)
    return [(w, t) for w, t in word_tag.items() if len(t) == m]


def tags_in_order_of_decreasing_frequency(bc):
    counts = collections.Counter((subl[1] for subl in bc.words))
    return counts.most_common(20)


def tags_that_nouns_are_most_commonly_found_after(bc):
    word_tag_pairs = nltk.bigrams(bc.words)
    # finds the tags that are NN , NNS , NNP or NNPS which represents nouns
    proceding_nouns = [a[1] for (a, b) in word_tag_pairs if
                       b[1] == "NN" or b[1] == "NNS" or b[1] == "NNP" or b[1] == "NNPS"]
    # puts into fd
    fd = nltk.FreqDist(proceding_nouns)
    # finds the most common
    return [(tag, v) for (tag, v) in fd.most_common()]


def proportion_ambiguous_word_types(bc):
    brown_tagged_words = bc.words
    cfd = nltk.ConditionalFreqDist(brown_tagged_words)
    conditions = cfd.conditions()
    mono_tags = [condition for condition in conditions if len(cfd[condition]) == 1]
    proportion_mono_tags = len(mono_tags) / len(conditions)

    return 1 - proportion_mono_tags


def proportion_ambiguous_word_tokens(bc):
    dict_tagged = {}
    for tuple in bc.words:
        word = tuple[0].lower()
        if word not in dict_tagged.keys():
            dict_tagged[word] = set()
        dict_tagged[word].add(tuple[1])

    fd = nltk.FreqDist([tag[0] for tag in bc.words])
    count = 0
    for word in dict_tagged.keys():
        if len(dict_tagged[word]) == 1:
            count += fd.get(word, 0)
    total = len(bc.words)
    return 1 - (count / total)
