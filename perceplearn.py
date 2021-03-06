import json
import time
import sys
import re
import string
from random import shuffle


def learn(ipfile):
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
                # shuffle(alllines)

                line = regex.sub(' ', i).split()
                # punc_list = list(string.punctuation)
                # for w in punc_list:
                #     i = i.replace(w, ' ')
                # line = i.split(' ')

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

                # print len(re.split(r'[.!?]+', i))
                # print line
                current_features['sent_size'] = len(re.split(r'[.!?]+', i))
                # raw_input()

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
    # tf_features = {x: y for x, y in tf_features.items() if y != 0}
    print len(tf_features)

    print "PN"
    print pn_bias
    # pn_features = {x: y for x, y in pn_features.items() if y != 0}
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
