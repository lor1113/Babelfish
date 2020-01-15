import random
import numpy as np
import copy
from nltk.corpus import wordnet
import operator as op
from functools import reduce
import json

markov = {}
start = []
length_mean = 0
eng_charset = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
results = [0.2, 2]
variance = 5.8
wordspace = 8031810176
length = 26
json_data = open('dictionary.json').read()
data = json.loads(json_data)
json_data2 = open('thesaurus.json').read()
thesaurus = json.loads(json_data2)

def writer(target,tbw):
    with open(target,'w') as outfile:
        json.dump(tbw,outfile)

def ncr(n, r):
    r = min(r, n - r)
    numer = reduce(op.mul, range(n, n - r, -1), 1)
    denom = reduce(op.mul, range(1, r + 1), 1)
    return numer / denom


def choose(dict):
    key = list(dict.keys())
    val = list(dict.values())
    return np.random.choice(key, None, True, val)


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def generate_length():
    json_data = open('lengths.json').read()
    lengths = json.loads(json_data)
    closest = find_nearest(lengths, length) / 1000
    closest = closest * random.uniform(0.75, 1.25)
    variance = 30 / length
    return closest, variance


def generate_markov(english):
    values = [0, 0]
    start = {}
    x = list(range(length))
    charset = []
    charset2 = []
    markov = {}
    if english == True:
        charset = copy.deepcopy(eng_charset)
        charset2 = copy.deepcopy(eng_charset)
        for each in eng_charset:
            start[each] = 0
    else:
        for i in range(length):
            charset.append("char" + str(i))
            charset2.append("char" + str(i))
            start["char" + str(i)] = 0

    def randomDistri(data):
        values[0] = data[0] * random.uniform(0.5, 1.5)
        values[1] = data[1] * random.uniform(0.5, 1.5)
        return values

    def y(x):
        j = x + 1
        return ((len(charset) + 1 - j) ** values[0]) / j ** values[1]

    for char1 in charset:
        markov[char1] = {}
        values = randomDistri(results)
        random.shuffle(charset2)
        ys = [y(j) for j in x]
        total = sum(ys)
        for j in x:
            markov[char1][charset2[j]] = y(j) / total

    for each in charset:
        char = choose(markov[each])
        for i in range(1000):
            char = choose(markov[char])
            start[char] = start[char] + 1
    sm = sum(start.values())
    start = {k: v / sm for (k, v) in start.items()}
    return markov, start


length_mean, variance = generate_length()
markov, start = generate_markov(True)
print(length_mean,variance)

words = list(data.keys())
words = list(set(words))
togo = len(words)
translated = []
translatepairs = {}
final = {}
leftover = []


def wordgen(word):
    try:
        base = wordnet.synsets(word)[0]
        synonyms = [i for i in thesaurus[word]["synonyms"] if i in translated]
        syns = random.sample(wordnet.synsets(word)[1:], round(len(wordnet.synsets(word)[1:]) / 2))
        leftover.append([item for item in wordnet.synsets(word)[1:] if item not in syns])
        if synonyms:
            nets = [[wordnet.synsets(i)[0], i] for i in synonyms]
            comp = 0
            best = 0
            for net in nets:
                similar = base.wup_similarity(net[0])
                if similar is None:
                    similar = 0
                if similar > best:
                    comp = net
                    best = similar
            if comp == 0:
                length = round(np.random.normal(length_mean, variance))
                output = choose(start)
                while len(output) < length:
                    output = output + choose(markov[output[-1]])
            else:
                word2 = translatepairs[comp[1]]
                rand = round(random.uniform(len(word2) / 2, len(word2)))
                if random.random() < 0.1:
                    rand = 0
                    output = choose(start)
                else:
                    output = word2[:rand]
                length = round(len(word2) * random.uniform(0.75, 1.25))
                while len(output) < length:
                    output = output + choose(markov[output[-1]])
        else:
            length = round(np.random.normal(length_mean, variance))
            output = choose(start)
            while len(output) < length:
                output = output + choose(markov[output[-1]])
        final[base._name] = output
        for each in syns:
            final[each._name] = output
    except:
        length = round(np.random.normal(length_mean, variance))
        output = choose(start)
        while len(output) < length:
            output = output + choose(markov[output[-1]])
        final[word] = output
    translated.append(word)
    translatepairs[word] = output
    return output

for word in words:
    if word.isalpha():
        wordgen(word.lower())
        togo = togo - 1
        if togo % 1000 == 0:
            print(word)
out = []
for each in leftover:
    if each:
        if isinstance(each, list):
            for each2 in each:
                out.append(each2._name)
        else:
            out.append(each._name)
out2 = []
for each in leftover:
    if each:
        if isinstance(each, list):
            for each2 in each:
                out2.append(each2)
        else:
            out2.append(each)
leftover = out2
togo = len(leftover)
for each in leftover:
    output = ""
    if each.hyponyms():
        length = round(np.random.normal(length_mean, variance))
        output = choose(start)
        while len(output) < length:
            output = output + choose(markov[output[-1]])
        final[each._name] = output
        for each in each.hyponyms():
            final[each._name] = output
    else:
        length = round(np.random.normal(length_mean, variance))
        output = choose(start)
        while len(output) < length:
            output = output + choose(markov[output[-1]])
        final[each._name] = output
    togo = togo - 1
    if togo % 1000 == 0:
        if output:
            print(output)

writer("final.json",final)