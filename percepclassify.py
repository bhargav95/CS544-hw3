import sys
import string
import json
import time
import re
import math


def classify(modelfile, ipfile):
    stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll",
                 "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's",
                 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs',
                 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is',
                 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did',
                 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at',
                 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',
                 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
                 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both',
                 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
                 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've",
                 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn',
                 "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't",
                 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn',
                 "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
    regex = re.compile('[%s]' % re.escape(string.punctuation))

    with open(ipfile) as f, open(modelfile) as f2, open("percepoutput.txt", "w") as op:

        data = json.load(f2)

        tf_features = data['TrueFake']['features']
        tf_bias = data['TrueFake']['bias']

        pn_features = data['PosNeg']['features']
        pn_bias = data['PosNeg']['bias']

        for i in f.readlines():

            line = regex.sub(' ', i).split()

            key = line[0]
            sent = line[1:]

            current_features = dict()
            current_features['sent_size'] = len(re.split(r'[.!?]+', i))

            for word in sent:
                lower = word.lower()

                if lower in stopwords:
                    continue

                if lower in current_features:
                    current_features[lower] += 1
                else:
                    current_features[lower] = 1.0

            tf_activation = tf_bias
            pn_activation = pn_bias
            for k, v in current_features.iteritems():

                if k in tf_features:
                    tf_activation += tf_features[k] * v

                if k in pn_features:
                    pn_activation += pn_features[k] * v

            if tf_activation < 0:
                class1 = "Fake"
            else:
                class1 = "True"

            if pn_activation < 0:
                class2 = "Neg"
            else:
                class2 = "Pos"

            op.write(" ".join([key, class1, class2 + "\n"]))


if __name__ == "__main__":
    assert len(sys.argv) == 3

    start = time.time()

    classify(sys.argv[1], sys.argv[2])