import copy
import json
import operator as op
import os
import time
from datetime import datetime
from functools import reduce
from pathlib import Path

import numpy as np
from nltk import pos_tag
from nltk.corpus import wordnet

lnumbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
eng_charset = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
disambiguate = ["JJ", "JJR", "JJS", "RB", "RBR", "RBS", "NN", "NNS", "VB", "VBG", "VBD", "VBN", "VBP", "VBZ"]
standin = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "A", "B",
           "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X"]
proper = ["NNP", "NNPS"]
results = [0.15, 1.75]
wordspace = 8031810176
length = 26
shiftspace = {
    2: 20,
    3: 79,
    4: 250,
    5: 791,
    6: 2500,
    7: 7906,
    8: 25000
}
json_data = open('thesaurus.json').read()
thesaurus = json.loads(json_data)
words = list(thesaurus.keys())
words = list(set(words))
translated = []
translatepairs = {}
final = {}


def gen_charset(lnum):
    charset = []
    set1 = [0, 1]
    set2 = [0, 1, 2, 3]
    for i in range(50):
        choice1 = np.random.choice(set1)
        choice2 = np.random.choice(set2)
        charset.append("L" + str(lnum) + str(i) + str(choice1) + str(choice2))
    print(charset)
    return (charset)


def writer(target, tbw):
    with open(target, 'w') as outfile:
        json.dump(tbw, outfile)


def ncr(n, r):
    r = min(r, n - r)
    numer = reduce(op.mul, range(n, n - r, -1), 1)
    denom = reduce(op.mul, range(1, r + 1), 1)
    return numer / denom


def choose(dict):
    key = list(dict.keys())
    val = list(dict.values())
    return np.random.choice(key, None, True, val)


def randomDistri(data, values):
    values[0] = data[0] * np.random.uniform(0.5, 1.5)
    values[1] = data[1] * np.random.uniform(0.5, 1.5)
    return values


def regexfit(strang, set):
    for a, b in zip(strang, set):
        if a != '*' and b != '*' and a != b:
            return False
    return True


def y(x, charset, values):
    j = x + 1
    return ((len(charset) + 1 - j) ** values[0]) / j ** values[1]


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def generate_length():
    json_data = open('lengths.json').read()
    lengths = json.loads(json_data)
    closest = find_nearest(lengths, length) / 1000
    closest = closest * np.random.uniform(0.75, 1.25)
    variance = 30 / length
    return closest, variance


def gen(start, basic, markov, length_mean, variance):
    length = round(np.random.normal(length_mean, variance))
    output = choose(start)
    output = output + choose(basic[output[0]])
    while len(output) < length:
        output = output + choose(markov[output[-1]][output[-2]])
    return output


def generate_markov(english):
    values = [0, 0]
    start = {}
    markov1 = {}
    x = list(range(length))
    markov = {}
    if english == True:
        charset = copy.deepcopy(eng_charset)
        charset2 = copy.deepcopy(eng_charset)
        charset3 = copy.deepcopy(eng_charset)
        for each in eng_charset:
            start[each] = 0
    else:
        charset = standin
        charset2 = copy.deepcopy(charset)
        charset3 = copy.deepcopy(charset)
        for each in charset:
            start[each] = 0
    np.random.shuffle(charset3)
    for h in charset3:
        for char1 in charset:
            markov[char1] = {}
            values = randomDistri(results, values)
            np.random.shuffle(charset2)
            ys = [y(j, charset, values) for j in x]
            total = sum(ys)
            for j in x:
                markov[char1][charset2[j]] = y(j, charset, values) / total

        for each in charset:
            char = choose(markov[each])
            for i in range(1000):
                char = choose(markov[char])
                start[char] = start[char] + 1
        markov1[h] = markov
    for char1 in charset:
        markov[char1] = {}
        values = randomDistri(results, values)
        np.random.shuffle(charset2)
        ys = [y(j, charset, values) for j in x]
        total = sum(ys)
        for j in x:
            markov[char1][charset2[j]] = y(j, charset, values) / total

    for each in charset:
        char = choose(markov[each])
        for i in range(1000):
            char = choose(markov[char])
            start[char] = start[char] + 1
    basic = markov
    sm = sum(start.values())
    start = {k: v / sm for (k, v) in start.items()}
    return markov1, start, basic, charset


def wordgen(word, start, basic, markov, leftover, length_mean, variance):
    tag = pos_tag([word])[0][1]
    if tag in proper:
        output = gen(start, basic, markov, length_mean, variance)
        final[word] = output
    else:
        try:
            base = wordnet.synsets(word)[0]
            synonyms = [i for i in thesaurus[word]["synonyms"] if i in translated]
            syns = np.random.choice(wordnet.synsets(word)[1:], round(len(wordnet.synsets(word)[1:]) / 2))
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
                    output = gen(start, basic, markov, length_mean, variance)
                else:
                    word2 = translatepairs[comp[1]]
                    rand = round(np.random.uniform(len(word2) / 2, len(word2)))
                    if np.random.random() < 0.1:
                        output = choose(start)
                        output = output + choose(basic[output[0]])
                    else:
                        output = word2[:rand]
                    length = round(len(word2) * np.random.uniform(0.75, 1.25))
                    while len(output) < length:
                        output = output + choose(markov[output[-1]][output[-2]])
            else:
                output = gen(start, basic, markov, length_mean, variance)
                final[base._name] = output
                for each in syns:
                    final[each._name] = output
        except:
            output = gen(start, basic, markov, length_mean, variance)
            final[word] = output
    translated.append(word)
    translatepairs[word] = output
    return output


def cycle(start, basic, markov, leftover, length_mean, variance):
    togo = len(words)
    for word in words:
        if word.isalpha():
            wordgen(word.lower(), start, basic, markov, leftover, length_mean, variance)
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
        if each.hyponyms():
            output = gen(start, basic, markov, length_mean, variance)
            final[each._name] = output
            for each in each.hyponyms():
                final[each._name] = output
        else:
            output = gen(start, basic, markov, length_mean, variance)
            final[each._name] = output
        togo = togo - 1
        if togo % 1000 == 0:
            if output:
                print(output)


def shiftgen():
    baseset = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, "*"]
    shiftset = {}
    keys = list(shiftspace.keys())
    togo = sum(shiftspace.values())
    print(togo)
    for i in range(min(keys), max(keys) + 1):
        print(i)
        shiftset[i] = {}
        shiftset[i]["0"] = 0
        for j in range(shiftspace[i]):
            num = 0
            num2 = 0
            while num == num2:
                num = np.random.choice(baseset, size=i)
                num = "".join(num)
                numset = [i for i in str(num)]
                while any((regexfit(num, x) for x in shiftset[i].keys())):
                    num = np.random.choice(baseset, size=i)
                    num = "".join(num)
                    numset = [i for i in str(num)]
                if len(str(num)) < i:
                    for k in range(i - len(str(num))):
                        numset.append(0)
                num2 = np.random.choice(numset, size=i, replace=False)
                num2 = "".join(num2)
            shiftset[i][num] = num2
            togo = togo - 1
            if togo % 1000 == 0:
                print([num, num2])
        shiftset[i].pop("0")
    return shiftset


def language(seed="", state="", english=True):
    starting = time.time()
    if seed:
        np.random.seed(seed)
    if state:
        state = ['MT19937', state, 623, 0, 0.0]
        np.random.set_state(state)
    else:
        np.random.seed()
    state = np.random.get_state()
    length_mean, variance = generate_length()
    print(length_mean, variance)
    markov, start, basic, charset = generate_markov(english)
    cycle(start, basic, markov, [], length_mean, variance)
    shiftset = shiftgen()
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    ending = time.time()
    data = {"Word Length": length_mean,
            "Length Variance": variance,
            "Charset Length": length,
            "Charset": charset,
            "Time Generated": dt_string,
            "Time to Generate": ending - starting}
    name = state[1][623]
    folder = Path("Languages/" + str(name))
    os.mkdir(folder)
    writer(folder / "final.json", final)
    writer(folder / "markov.json", markov)
    writer(folder / "basic.json", basic)
    writer(folder / "start.json", start)
    writer(folder / "shiftset.json", shiftset)
    writer(folder / "state.json", state[1].tolist())
    writer(folder / "data.json", data)


def gameLanguage(name, lnum):
    starting = time.time()
    np.random.seed()
    mapper = {}
    chararr = gen_charset(lnum)
    for i in range(len(chararr)):
        mapper[standin[i]] = chararr[i]
    print(mapper)
    state = np.random.get_state()
    length_mean, variance = generate_length()
    print(length_mean, variance)
    markov, start, basic, charset = generate_markov(False)
    cycle(start, basic, markov, [], length_mean, variance)
    shiftset = shiftgen()
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    ending = time.time()
    data = {"Word Length": length_mean,
            "Length Variance": variance,
            "Charset Length": length,
            "Charset": charset,
            "Time Generated": dt_string,
            "Time to Generate": ending - starting}
    folder = Path("Data/Languages/" + str(name))
    os.mkdir(folder)
    writer(folder / "final.json", final)
    writer(folder / "markov.json", markov)
    writer(folder / "basic.json", basic)
    writer(folder / "start.json", start)
    writer(folder / "shiftset.json", shiftset)
    writer(folder / "mapper.json", mapper)
    writer(folder / "state.json", state[1].tolist())
    writer(folder / "data.json", data)


def gamegen():
    starting = time.time()
    chosen = np.random.choice(lnumbers, 7)
    for i in range(len(chosen)):
        gameLanguage(i, chosen[i])
    ending = time.time()
    print(ending - starting)


gamegen()
