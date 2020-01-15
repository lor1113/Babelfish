import json
from nltk.corpus import wordnet
dict = {}
def writer(target,tbw):
    with open(target,'w') as outfile:
        json.dump(tbw,outfile)

words = list(wordnet.words())
for word in words:
    dict[word] = 0

writer("Dictionary.json",dict)