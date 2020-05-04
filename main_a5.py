import sys

from Bayes import Bayes
from DecisionTree import DecisionTree

if __name__ == "__main__":

    if sys.argv[1]== "--train":
        train = Bayes(False)
        dt = DecisionTree(False)
        train.raw_count()
        train.create_binary_trainer()
        train.negation_create()
        train.create_senti()
        train.dump_to_file()
        dt.raw_count()
        dt.create_binary_trainer()
        dt.negation_create()
        dt.create_senti()
        dt.dump_to_file()

    else:
        train = None
        if sys.argv[2] == "tree":
            train = DecisionTree(True)
        else:
            train = Bayes(True)
        user = 0

        while user <= 0 or user > 4:
            print("Choose a model:\n"
                  "1 - all words raw counts\n"
                  "2 - all words binary\n"
                  "3 - SentiWordNet words\n"
                  "4 - all words plus Negation"
                  )
            try:
                user = int(input("Type a number:" ))
            except ValueError:
                print("invalid input try again.")
                continue
        file = sys.argv[3]
        if user == 1:
            print(train.get_raw(file))
        elif user == 2:
            print(train.get_pos_neg(file))
        elif user == 3:
            print(train.get_senti(file))
        elif user == 4:
            print(train.get_neg(file))




