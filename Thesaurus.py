import json
import nltk
from nltk.corpus import wordnet

def writer(target,tbw):
    with open(target,'w') as outfile:
        json.dump(tbw,outfile)


json_data = open('data.json').read()
data = json.loads(json_data)
thesaurus = {}
wordlist = list(wordnet.words())
togo = len(wordlist)
for word in wordlist:
    synonyms = []
    antonyms = []
    tag = nltk.pos_tag([word])[0][1]
    clas = data["classes"][tag]
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            if l.name() not in synonyms:
                synonyms.append(l.name())
            if l.antonyms():
                if l.name() not in antonyms:
                    antonyms.append(l.antonyms()[0].name())
    thesaurus[word] = {
        "Tag": tag,
        "Class": clas,
        "synonyms": synonyms,
        "antonyms": antonyms,
    }
    togo = togo - 1
    if togo % 1000 == 0:
        print(togo)
        print(thesaurus[word])

writer("thesaurus.json",thesaurus)

