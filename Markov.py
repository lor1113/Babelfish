import json
from nltk.corpus import brown
charset = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]
markov = {}

def writer(target,tbw):
    with open(target,'w') as outfile:
        json.dump(tbw,outfile)

for each in charset:
    markov[each] = {}

for each1 in charset:
    for each2 in charset:
        markov[each1][each2] = 0

print(markov)
words = list(brown.words())
for word in words:
    res = [ele for ele in charset if (ele in word)]
    if bool(res):
        word = word.lower()
        word = ''.join(filter(charset.__contains__, word))
        for i in range(len(word)):
            if i < len(word)-1:
                markov[word[i]][word[i+1]] = markov[word[i]][word[i+1]] + 1

for each in markov:
    print(sum(markov[each].values()))

print(markov)
writer("markov_raw.json",markov)