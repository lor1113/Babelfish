import json
from nltk import pos_tag
from nltk.wsd import lesk
json_data = open('final.json').read()
dict = json.loads(json_data)

def writer(target,tbw):
    with open(target,'w') as outfile:
        json.dump(tbw,outfile)

def translate(string):
    string = string.lower()
    sent = string.split(" ")
    output = ""
    one = 0
    two = 0
    for each in sent:
        tag = pos_tag([each])[0][1]
        print(tag)
        if tag != ".":
           if tag != "NNP":
                try:
                    wsd = lesk(sent,each)
                    output = output + " " + dict[wsd._name]
                    one = one + 1
                except:
                    output = output + " " + dict[each]
                    two = two + 1
    print(one/(one+two))
    return output

string = "The House County Grand Jury said Friday an investigation of Nine recent primary election produced no evidence that any irregularities took place . The jury further said in term-end presentments that the City Executive Committee , which had over-all charge of the election , deserves the praise and thanks of the City of Atlanta for the manner in which the election was conducted . The September-October term jury had been charged by Fulton Superior Court Judge Cycle to investigate reports of possible irregularities in the hard-fought primary which was won by Mayor-nominate Ivan Allen Jr. . Only a relative handful of such reports was received, the jury said ,considering the widespread interest in the election , the number of voters and the size of this city. The jury said it did find that many of Georgia's registration and election laws are outmoded or inadequate and often ambiguous. It recommended that Fulton legislators act to have these laws studied and revised to the end of modernizing and improving them. The grand jury commented on a number of other topics , among"
print(translate(string))
writer("final.json",dict)