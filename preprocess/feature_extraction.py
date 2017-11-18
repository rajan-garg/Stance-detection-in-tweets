import os
from config import DIC_FILE, AFF_FILE, DATA_FILE, TARGETS_PATH, CSV_PATH, dummy_stop_words
from text_helper import TextHelper
import csv


class FeatureExtraction:

    def __init__(self, textHelper):
        self.textHelper = textHelper
        self.tweet_id = 1

    def run(self):
        target_files = [f for f in os.listdir(TARGETS_PATH) if os.path.isfile(os.path.join(TARGETS_PATH, f))]
        for target_file in target_files:
            self.process_file(target_file)

    def process_file(self, target_file):

        print("Processing target: " + target_file)

        print("Reading tweets...")

        data = self.read_tweets(target_file)

        print("Generating n-grams...")

        _1_gram, _2_gram, _3_gram = self.generate_n_grams(target_file, data)

        print("Extracting features...")
        
        fieldnames = self.extract_features(_1_gram, _2_gram, _3_gram)

        print("#features = " + str(len(fieldnames)))

        with open(CSV_PATH + target_file + '.csv', 'w+') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()      
            progress = 0
            for d in data:
                meta_data = {}
                meta_data["ID"] = d["ID"]
                meta_data["LABEL"] = d["LABEL"]
                meta_data["TARGET"] = target_file
                zero_row = dict(meta_data)
                zero_row.update(_1_gram)
                zero_row.update(_2_gram)
                zero_row.update(_3_gram)

                row_features = self.features(d, zero_row)
                writer.writerow(row_features)

                progress = progress + 1
                print(target_file + ": " + str(progress*100/len(data)) + " % Done")

    def read_tweets(self, target_file):
        data = []
        with open(TARGETS_PATH + "/" + target_file) as f:
            for line in f:
                line = line.lower()
                d = {}
                d["LABEL"] = line[0]
                if line[0] == '-':
                    d["LABEL"] = d["LABEL"] + line[1]
                d["tweet"] = self.textHelper.lemmatize(line[line.index(target_file.lower()) + len(target_file):])
                d["ID"] = self.tweet_id
                data.append(d)
                self.tweet_id = self.tweet_id + 1
        return data

    def generate_n_grams(self, target_file, data):
        _1_gram = {}
        _2_gram = {}
        _3_gram = {}
        for d in data:
            words = self.textHelper.tokenize(d["tweet"])
            for word in words:
                try:
                    _1_gram[word] = 0
                except KeyError:
                    print(word)
            for i in range(0, len(words)-1):
                try:
                    _2_gram[str(words[i] + "_" + words[i+1])] = 0
                except KeyError:
                    print(words[i] + "_" + words[i+1])
            for i in range(0, len(words)-2):
                try:
                    _3_gram[str(words[i] + "_" + words[i+1] + "_" + words[i+2])] = 0
                except KeyError:
                    print(words[i] + "_" + words[i+1] + "_" + words[i+2])

        return _1_gram, _2_gram, _3_gram

    def extract_features(self, _1_gram, _2_gram, _3_gram):
        fieldnames = ['ID', 'LABEL', 'TARGET']
        for key, value in _1_gram.items():
            fieldnames.append(key)
        for key, value in _2_gram.items():
            fieldnames.append(key)
        for key, value in _3_gram.items():
            fieldnames.append(key)
        return fieldnames

    def features(self, d, zero_row):
        words = self.textHelper.tokenize(d["tweet"])
        for word in words:
            try:
                zero_row[word] = 1
            except KeyError:
                print(word)
        for i in range(0, len(words)-1):
            try:
                zero_row[str(words[i] + "_" + words[i+1])] = 1
            except KeyError:
                print(words[i] + "_" + words[i+1])
        for i in range(0, len(words)-2):
            try:
                zero_row[str(words[i] + "_" + words[i+1] + "_" + words[i+2])] = 1
            except KeyError:
                print(words[i] + "_" + words[i+1] + "_" + words[i+2])
        return zero_row


if __name__ == '__main__':

    textHelper = TextHelper(DIC_FILE, AFF_FILE, dummy_stop_words)

    featureExtraction = FeatureExtraction(textHelper)

    featureExtraction.run()

