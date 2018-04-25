import sys
import string
import json
import time
import re
import math


def classify(modelfile, ipfile):
    stopwords = {"ourselves", "hers", "between", "yourself", "but", "again", "there", "about", "once", "during", "out",
                 "very", "having", "with", "they", "own", "an", "be", "some", "for", "do", "its", "yours", "such",
                 "into", "of", "most", "itself", "other", "off", "is", "s", "am", "or", "who", "as", "from", "him",
                 "each", "the", "themselves", "until", "below", "are", "we", "these", "your", "his", "through", "don",
                 "nor", "me", "were", "her", "more", "himself", "this", "down", "should", "our", "their", "while",
                 "above", "both", "up", "to", "ours", "had", "she", "all", "no", "when", "at", "any", "before", "them",
                 "same", "and", "been", "have", "in", "will", "on", "does", "yourselves", "then", "that", "because",
                 "what", "over", "why", "so", "can", "did", "not", "now", "under", "he", "you", "herself", "has",
                 "just", "where", "too", "only", "myself", "which", "those", "i", "after", "few", "whom", "t", "being",
                 "if", "theirs", "my", "against", "a", "by", "doing", "it", "how", "further", "was", "here", "than"}
    regex = re.compile('[%s]' % re.escape(string.punctuation))

    with open(ipfile) as f, open(modelfile) as f2, open("percepoutput.txt", "w") as op:

        data = json.load(f2)

        tf_features = data['TrueFalse']['features']
        tf_bias = data['TrueFalse']['bias']

        pn_features = data['PosNeg']['features']
        pn_bias = data['PosNeg']['bias']

        for i in f.readlines():

            line = regex.sub(' ', i).split()
            key = line[0]
            sent = line[1:]

            current_features = dict()

            for word in sent:
                lower = word.lower()

                if lower in stopwords:
                    continue

                if lower in current_features:
                    current_features[lower] += 1
                else:
                    current_features[lower] = 0.0

            tf_activation = tf_bias
            pn_activation = pn_bias
            for k, v in current_features.iteritems():

                if k in tf_features:
                    tf_activation += tf_features[k]

                if k in pn_features:
                    pn_activation += pn_features[k]

            if tf_activation < 0:
                class1 = "False"
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
