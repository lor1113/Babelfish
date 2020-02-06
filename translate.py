import json
import string
from pathlib import Path

import nltk
import numpy as np
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from pywsd.lesk import simple_lesk

name = 4285349879
folder = Path("Languages/" + str(name))
json_data = open(folder / 'final.json').read()
dict = json.loads(json_data)
json_data = open(folder / 'markov.json').read()
markov = json.loads(json_data)
json_data = open(folder / 'basic.json').read()
basic = json.loads(json_data)
json_data = open(folder / 'shiftset.json').read()
shiftset = json.loads(json_data)
json_data = open(folder / 'start.json').read()
start = json.loads(json_data)
json_data = open(folder / 'data.json').read()
data = json.loads(json_data)
disambiguate = ["JJ", "JJR", "JJS", "RB", "RBR", "RBS", "NN", "NNS", "VB", "VBG", "VBD", "VBN", "VBP", "VBZ"]
proper = ["NNP", "NNPS"]
length_mean = data['Word Length']
variance = data["Length Variance"]
punctuation = []
np.random.seed(name)
tagnums = {
    "ADJ": 0,
    "ADP": 1,
    "ADV": 2,
    "CONJ": 3,
    "DET": 4,
    "NOUN": 5,
    "NUM": 6,
    "PRT": 7,
    "PRON": 8,
    "VERB": 9
}


def digitsum(x):
    total = 0
    for letter in str(x):
        total = total + ord(letter)
    return total


def digitalt(x):
    total = 1
    tick = 0
    for letter in str(x):
        if tick == 0:
            total = total + ord(letter)
            tick = tick + 1
        elif tick == 0:
            total = total * ord(letter)
            tick = tick + 1
        elif tick == 0:
            total = total - ord(letter)
            tick = tick + 1
        elif tick == 0:
            total = total / ord(letter)
            tick = 0
    return (round(total))


def choose(dict):
    key = list(dict.keys())
    val = list(dict.values())
    return np.random.choice(key, None, True, val)


def writer(target, tbw):
    with open(target, 'w') as outfile:
        json.dump(tbw, outfile)


def shuffle(sentence):
    letters = "".join([x for x in sentence if x in string.ascii_letters])
    seed = digitalt(letters) * digitsum(letters)
    np.random.seed(seed)
    sprinkle = []
    sent = word_tokenize(sentence)
    tags = pos_tag(sent, tagset="universal")
    inter = ""
    output = ""
    tick = 0
    for i in range(len(tags)):
        if tags[i][1] == ".":
            sent.remove(tags[i][0])
            if tags[i][0] not in punctuation:
                punctuation.append(tags[i][0])
            if i == len(tags) - 1:
                end = tags[i][0]
            else:
                sprinkle.append(tags[i][0])
        else:
            sprinkle.append(0)
            num = tagnums[tags[i][1]]
            inter = inter + str(num)
    for i in range(9, 1, -1):
        if not i > len(inter):
            mod = len(inter) % i
            if mod == 0:
                splits = [inter[k:k + i] for k in range(0, len(inter), i)]
                shift = 0
            else:
                if np.random.random() < 0.5:
                    inter2 = inter[:-1 * mod]
                    splits = [inter2[k:k + i] for k in range(0, len(inter2), i)]
                    shift = 0
                else:
                    inter2 = inter[mod:]
                    splits = [inter2[k:k + i] for k in range(0, len(inter2), i)]
                    shift = mod
            for j in range(len(splits)):
                intset = [[], [], [], [], [], [], [], [], [], [], []]
                each = splits[j]
                if each in shiftset[str(i)]:
                    target = shiftset[str(i)][each]
                    pos = shift + ((j) * i)
                    slice = sent[pos:pos + i]
                    for l in range(len(str(each))):
                        intset[int(str(each)[l])].append(slice[l])
                    for m in range(len(str(target))):
                        f = np.random.choice(intset[int(str(target)[m])])
                        intset[int(str(target)[m])].remove(f)
                        sent[pos] = f
                        pos = pos + 1
    for each in sprinkle:
        if each == 0:
            output = output + sent[tick] + " "
            tick = tick + 1
        else:
            output = output.strip() + each + " "
    return output.strip()


def gen(seed):
    np.random.seed(seed)
    length = round(np.random.normal(length_mean, variance))
    output = choose(start)
    output = output + choose(basic[output[0]])
    while len(output) < length:
        output = output + choose(markov[output[-1]][output[-2]])
    return output


def sentence_translate(strang):
    sent = word_tokenize(strang)
    sprinkle = []
    end = ""
    tags = pos_tag(sent, tagset="universal")
    for i in range(len(tags)):
        if tags[i][1] == ".":
            sent.remove(tags[i][0])
            if tags[i][0] not in punctuation:
                punctuation.append(tags[i][0])
            if i == len(tags) - 1:
                end = tags[i][0]
            else:
                sprinkle.append(tags[i][0])
        else:
            sprinkle.append(0)
    output = ""
    for i in range(len(sent)):
        each = sent[i]
        each = "".join([x for x in each if x in string.ascii_letters])
        punct = ""
        if i < len(sent) - 1:
            if sprinkle[0] == 0:
                sprinkle.pop(0)
                if np.random.random() < .05:
                    punct = np.random.choice(punctuation)
            else:
                if np.random.random() < .25:
                    sprinkle.pop(0)
                else:
                    punct = sprinkle[0]

        tag = pos_tag([each])[0][1]
        if tag not in proper:
            if each in dict.keys():
                output = output + " " + dict[each] + punct
            else:
                word = gen(digitsum(each) * digitalt(each))
                output = output + " " + word + punct
                dict[each] = word
        else:
            try:
                wsd = simple_lesk(string, each)
                output = output + " " + dict[wsd._name]
            except:
                if each in dict.keys():
                    output = output + " " + dict[each] + punct
                else:
                    word = gen(digitsum(each) * digitalt(each))
                    output = output + " " + word + punct
                    dict[each] = word
    return output + end


def translate(stringinput):
    if all(c in string.printable for c in stringinput):
        output = ""
        sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
        sentences = sent_detector.tokenize(stringinput.strip())
        for sentence in sentences:
            sentence = shuffle(sentence)
            output = output + " " + sentence_translate(sentence)
        output = " ".join(output.split())
        return output
    else:
        print("Input does not entirely consist of ASCII Characters. Offending characters were:")
        print([c for c in stringinput if c not in string.printable])


stringinput = "Hello there, general Kenobi"
print(translate(stringinput))
writer(folder / "final.json", dict)
