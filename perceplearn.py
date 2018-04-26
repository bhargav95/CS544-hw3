import json
import time
import sys
import re
import string
from random import shuffle


def learn(ipfile):
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

    tf_features = dict()
    tf_bias = 0.0
    tfu_bias = 0.0
    tfu_features = dict()

    pn_features = dict()
    pn_bias = 0.0
    pnu_bias = 0.0
    pnu_features = dict()

    c = 1.0

    with open(ipfile) as f:
        alllines = f.readlines()

        for iteri in range(30):
            for i in alllines:
                shuffle(alllines)

                line = regex.sub(' ', i).split()
                # line = i.translate(None, string.punctuation).strip().split()
                # print i
                # print line
                # raw_input()

                # id = line[0]
                class1 = line[1]
                class2 = line[2]
                sent = line[3:]

                tf_y = {"Fake": -1, "True": 1}
                pn_y = {"Neg": -1, "Pos": 1}

                current_features = dict()

                for word in sent:
                    lower = word.lower()

                    if lower in stopwords:
                        continue

                    if lower in current_features:
                        current_features[lower] += 1
                    else:
                        current_features[lower] = 1.0

                # Activation
                tf_activation = tf_bias
                pn_activation = pn_bias
                for k, v in current_features.iteritems():

                    if k in tf_features:
                        tf_activation += tf_features[k] * v

                    if k in pn_features:
                        pn_activation += pn_features[k] * v

                # TF Update
                if tf_y[class1] * tf_activation <= 0:
                    tf_bias += tf_y[class1]
                    tfu_bias += tf_y[class1] * c
                    for k, v in current_features.iteritems():

                        if k in tfu_features:
                            tfu_features[k] += tf_y[class1] * c * v
                        else:
                            tfu_features[k] = tf_y[class1] * c * v

                        if k in tf_features:
                            tf_features[k] += tf_y[class1] * v
                        else:
                            tf_features[k] = tf_y[class1] * v

                # PN Update
                if pn_y[class2] * pn_activation <= 0:
                    pn_bias += pn_y[class2]
                    pnu_bias += pn_y[class2] * c
                    for k, v in current_features.iteritems():

                        if k in pnu_features:
                            pnu_features[k] += pn_y[class2] * c * v
                        else:
                            pnu_features[k] = pn_y[class2] * c * v

                        if k in pn_features:
                            pn_features[k] += pn_y[class2] * v
                        else:
                            pn_features[k] = pn_y[class2] * v

                c += 1

    print "TF"
    print tf_bias
    tf_features = {x: y for x, y in tf_features.items() if y != 0}
    print len(tf_features)

    print "PN"
    print pn_bias
    pn_features = {x: y for x, y in pn_features.items() if y != 0}
    print len(pn_features)

    averagedtf_features = dict()
    averagedtf_bias = tf_bias - (tfu_bias / c)

    averagedpn_features = dict()
    averagedpn_bias = pn_bias - (pnu_bias / c)

    for k, v in tf_features.iteritems():
        averagedtf_features[k] = tf_features[k] - (tfu_features[k] / c)

    for k, v in pn_features.iteritems():
        averagedpn_features[k] = pn_features[k] - (pnu_features[k] / c)

    with open("averagedmodel.txt", "w") as op:
        json.dump({"TrueFake": {"features": averagedtf_features, "bias": averagedtf_bias},
                   "PosNeg": {"features": averagedpn_features, "bias": averagedpn_bias}}, op, indent=1)

    with open("vanillamodel.txt", "w") as op:
        json.dump({"TrueFake": {"features": tf_features, "bias": tf_bias},
                   "PosNeg": {"features": pn_features, "bias": pn_bias}}, op, indent=1)


if __name__ == "__main__":
    assert len(sys.argv) == 2

    start = time.time()

    learn(sys.argv[1])

    print time.time() - start
