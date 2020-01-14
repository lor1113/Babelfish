import random
import numpy as np
import copy
import operator as op
import math
from functools import reduce
import json
markov = {}
start = []
eng_charset = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
results = [0.2, 2]
wordspace = 8031810176
lengths = [8031810176.0, 89620.36697090679, 2002.6473422433817, 299.36660964594364, 95.71118365501964, 44.750947947986326, 25.999999999999996, 17.30221400994519, 12.604767118321744, 9.783209271758405, 7.951289848172858, 6.689614932713716, 5.779745637247233, 5.0990195135927845, 4.574260521766692, 4.159593010132745, 3.825040910262351, 3.5503192980803493, 3.3212836710738927]
length = 26


def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
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
    closest = find_nearest(lengths,length) + 1
    print(closest)


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
            start[char]= start[char] + 1
    sm = sum(start.values())
    start = {k:v/sm for (k,v) in start.items()}
    return markov,start

generate_length()