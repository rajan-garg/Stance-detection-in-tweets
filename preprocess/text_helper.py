import re
from nltk.stem.wordnet import WordNetLemmatizer
from hunspell import HunSpell


class TextHelper:

    def __init__(self, DIC_FILE, AFF_FILE, dummy_stop_words):
        self.__dummy_stop_words = dummy_stop_words
        self.__lmtzr = WordNetLemmatizer()
        self.__hunspell_object = HunSpell(DIC_FILE, AFF_FILE)

    def lemmatize(self, line):
        tokens = self.tokenize(line)
        lmtzd_line = ""
        for token in tokens:
            lmtzd_line = lmtzd_line + self.lemmatize_word(token) + " "
        return lmtzd_line

    def stem(self, line):
        tokens = self.tokenize(line)
        stemmed_line = ""
        for token in tokens:
            stemmed_line = stemmed_line + self.stem_word(token) + " "
        return stemmed_line

    def lemmatize_word(self, word):
        word1 = self.__lmtzr.lemmatize(word, 'n')
        word2 = self.__lmtzr.lemmatize(word, 'v')
        if word2 != word:
            return word2
        return word1

    def stem_word(self, word):
        stemmed_list = self.__hunspell_object.stem(word)
        if len(stemmed_list) > 0:
            return str(stemmed_list[0])
        else:
            return word

    def tokenize(self, text):
        return re.split('\W+' , text)

    def separate_targets(self):
        count = 0
        new_file = open("Surgical Strike", "w+")
        with open(DATA_FILE) as f:
            for line in f:
                count = count + 1
                if count>50000:
                    new_file.write(line)
                if "Surgical Strike" in line and "Uri Attack" not in line and "Pathankot Attack" not in line and "GSTN" not in line and  "Kashmir Unrest" not in line :
                    new_file.write(line)

