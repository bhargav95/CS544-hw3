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

    pn_features = dict()
    pn_bias = 0.0

    with open(ipfile) as f:
        alllines = f.readlines()

        for iter in range(20):
            for i in alllines:
                shuffle(alllines)
                line = regex.sub(' ', i).split()

                id = line[0]
                class1 = line[1]
                class2 = line[2]
                sent = line[3:]

                tf_y = {"Fake": -1, "True": 1}
                pn_y = {"Neg": -1, "Pos": 1}

                current_features = dict()

                for word in sent:
                    if word in current_features:
                        current_features[word] += 1
                    else:
                        current_features[word] = 0.0

                # print current_features

                # TF Activation
                tf_activation = tf_bias
                for k, v in current_features.iteritems():

                    if k in tf_features:
                        tf_activation += tf_features[k]

                # TF Update
                if tf_y[class1] * tf_activation <= 0:
                    tf_bias += tf_y[class1]
                    for k, v in current_features.iteritems():

                        # print tf_y[class1]
                        # print current_features[k]

                        if k in tf_features:
                            tf_features[k] += tf_y[class1] * current_features[k]
                        else:
                            tf_features[k] = tf_y[class1] * current_features[k]

                # PN Activation
                pn_activation = pn_bias
                for k, v in current_features.iteritems():

                    if k in pn_features:
                        pn_activation += pn_features[k]

                # pn Update
                if pn_y[class2] * pn_activation <= 0:
                    pn_bias += pn_y[class2]
                    for k, v in current_features.iteritems():

                        # print pn_y[class1]
                        # print current_features[k]

                        if k in pn_features:
                            pn_features[k] += pn_y[class2] * current_features[k]
                        else:
                            pn_features[k] = pn_y[class2] * current_features[k]

    print "TF"
    print tf_bias
    tf_features = {x: y for x, y in tf_features.items() if y != 0}
    print len(tf_features)

    print "PN"
    print pn_bias
    pn_features = {x: y for x, y in pn_features.items() if y != 0}
    print len(pn_features)


if __name__ == "__main__":
    assert len(sys.argv) == 2

    start = time.time()

    learn(sys.argv[1])

    print time.time()-start
