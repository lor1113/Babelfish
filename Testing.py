from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
import numpy as np
import json
charset = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]
json_data = open('markov.json').read()
markov = json.loads(json_data)

def test(x, a, b, c, d, e):
    return ((a*x**4) + (b*x**3) +(c*x**2) + (d*x) + e)

def writer(target,tbw):
    with open(target,'w') as outfile:
        json.dump(tbw,outfile)

fits = {}
x = list(range(1,27))
for each in markov:
    y=[]
    for each2 in markov:
        y.append(markov[each][each2])
    y = sorted(y)
    param, param_cov = curve_fit(test, x, y)
    fits[each] = param
    print(each)


writer("fits.json",fits)


