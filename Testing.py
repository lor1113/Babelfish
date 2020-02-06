import json

import numpy as np

length = 50


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


json_data = open('lengths.json').read()
lengths = json.loads(json_data)


def generate_length():
    closest = find_nearest(lengths, length) / 1000
    closest = closest * np.random.uniform(0.75, 1.25)
    variance = 30 / length
    return closest, variance


print(find_nearest(lengths, length))
print(generate_length())
print(len(lengths))
