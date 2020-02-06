import numpy as np

shiftspace = {
    2: 45,
    3: 214,
    4: 921,
    5: 3532,
    6: 11808,
    7: 33399,
    8: 76134,
    9: 128042
}

np.random.seed(99)


def shiftgen():
    baseset = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, "*"]
    shiftset = {}
    keys = list(shiftspace.keys())
    togo = sum(shiftspace.values())
    print(togo)
    for i in range(min(keys), max(keys) + 1):
        print(i)
        shiftset[i] = {}
        for j in range(shiftspace[i]):
            num = 0
            num2 = 0
            while num == num2:
                num = np.random.choice(baseset, size=i)
                num = "".join(num)
                numset = [i for i in str(num)]
                while num in list(shiftset[i].keys()):
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
            print([num, num2])
    return shiftset


print(shiftgen())
