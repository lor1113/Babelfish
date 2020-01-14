import math
wordspace = 8031810176
length = 26
lengths = []
for x in range(1,20,0.01):
    lengths.append(wordspace ** (1/x))