
#            _         _____          _                     
#  ___ _   _| |__   __|_   _| __ __ _(_)_ __    _ __  _   _ 
# / __| | | | '_ \ / _ \| || '__/ _` | | '_ \  | '_ \| | | |
#| (__| |_| | |_) |  __/| || | | (_| | | | | |_| |_) | |_| |
# \___|\__,_|_.__/ \___||_||_|  \__,_|_|_| |_(_) .__/ \__, |
#                                              |_|    |___/ 

# Provides general tools to help train
# AI Models.

import cube
import random
import numpy as np


# Create the inverse Dictionary
actionInverse = {
    'r' : '.r',
    'l' : '.l',
    'u' : '.u',
    'd' : '.d',
    'f' : '.f',
    'b' : '.b',
    '.r': 'r',
    '.l': 'l',
    '.u': 'u',
    '.d': 'd',
    '.f': 'f',
    '.b': 'b',
    ' ' : ' ',
}

# These are actions that when paired,
# will simply lead to the same state that
# they were in before the pair
# THIS IS ONLY TRUE FOR ORDER=2
actionAnti = {
    'r' : '.l',
    'l' : '.r',
    'u' : '.d',
    'd' : '.u',
    'f' : '.b',
    'b' : '.f',
    '.r': 'l',
    '.l': 'r',
    '.u': 'd',
    '.d': 'u',
    '.f': 'b',
    '.b': 'f',
}

# This the actions mapped to their
# Vectorized outputs
actionVector={
    'r' : [1,0,0,0,0,0,0,0,0,0,0,0],
    'l' : [0,1,0,0,0,0,0,0,0,0,0,0],
    'u' : [0,0,1,0,0,0,0,0,0,0,0,0],
    'd' : [0,0,0,1,0,0,0,0,0,0,0,0],
    'f' : [0,0,0,0,1,0,0,0,0,0,0,0],
    'b' : [0,0,0,0,0,1,0,0,0,0,0,0],
    '.r': [0,0,0,0,0,0,1,0,0,0,0,0],
    '.l': [0,0,0,0,0,0,0,1,0,0,0,0],
    '.u': [0,0,0,0,0,0,0,0,1,0,0,0],
    '.d': [0,0,0,0,0,0,0,0,0,1,0,0],
    '.f': [0,0,0,0,0,0,0,0,0,0,1,0],
    '.b': [0,0,0,0,0,0,0,0,0,0,0,1],
    ' ' : [0,0,0,0,0,0,0,0,0,0,0,0],
}

# Given an index, produce an output vector
indexToVector={
    0  : [1,0,0,0,0,0,0,0,0,0,0,0],
    1  : [0,1,0,0,0,0,0,0,0,0,0,0],
    2  : [0,0,1,0,0,0,0,0,0,0,0,0],
    3  : [0,0,0,1,0,0,0,0,0,0,0,0],
    4  : [0,0,0,0,1,0,0,0,0,0,0,0],
    5  : [0,0,0,0,0,1,0,0,0,0,0,0],
    6  : [0,0,0,0,0,0,1,0,0,0,0,0],
    7  : [0,0,0,0,0,0,0,1,0,0,0,0],
    8  : [0,0,0,0,0,0,0,0,1,0,0,0],
    9  : [0,0,0,0,0,0,0,0,0,1,0,0],
    10 : [0,0,0,0,0,0,0,0,0,0,1,0],
    11 : [0,0,0,0,0,0,0,0,0,0,0,1],
}

# Given an output vector, you can
# find the corresponding action
indexToAction={
    0  : 'r',
    1  : 'l',
    2  : 'u',
    3  : 'd',
    4  : 'f',
    5  : 'b',
    6  : '.r',
    7  : '.l',
    8  : '.u',
    9  : '.d',
    10 : '.f',
    11 : '.b',
}

# Gods number 
numGod = {
    2:14,
    3:20
}

# Creates a batched dataset
def ncubeCreateBatchMLN(batch_size, depth, orderNum):
    x_batch=[]
    y_batch=[]
    scrambles = generateScrambles(scramble_size=batch_size,max_len=depth,token='BALANCED', orderNum=orderNum)
    for scram in scrambles:
        ncube = cube.Cube(order=orderNum)
        for action in scram:
            ncube.minimalInterpreter(action)
        x_batch.append(ncube.constructVectorState(inBits=True))
        y_batch.append(actionVector[actionInverse[scram[-1]]])
    return np.array(x_batch,dtype='float32'), np.array(y_batch,dtype='float32')


# Create a batched dataset for LSTM
def ncubeCreateBatchLSTM(batch_size, depth, orderNum):
    x_batch=[]
    y_batch=[]
    scrambles = generateScrambles(scramble_size=batch_size, max_len=depth, token='FIXED', orderNum=orderNum)
    for scram in scrambles:
        #print(scram)
        y_temp = []
        x_temp = []
        ncube = cube.Cube(order=orderNum)
        for k in scram:
            ncube.minimalInterpreter(k)
            x_temp.append(ncube.constructVectorState(inBits=True))
            y_temp.append(actionVector[actionInverse[k]])
        x_batch.append(x_temp[:])
        y_batch.append(y_temp[:])
    return np.array(x_batch,dtype='float32'), np.array(y_batch,dtype='float32')


# Generate Random Sized Scrambles from a fixed scramble size
def ncubeCreateBatchSRNN(batch_size, depth, orderNum):
    x_batch=[]
    y_batch=[]
    scrambles = generateScrambles(scramble_size=batch_size, max_len=depth,token='FIXED', orderNum=orderNum)
    for count in range(batch_size):
        scrambles[count] = scrambles[count] + [' ']
        ncube = cube.Cube(order=orderNum)
        mydepth = (count % depth) + 1
        for i in range(mydepth):
            ncube.minimalInterpreter(scrambles[count][i])
        previousMove = actionVector[actionInverse[scrambles[count][mydepth]]]
        x_batch.append(ncube.constructVectorState(inBits=True)+previousMove)
        y_batch.append(actionVector[actionInverse[scrambles[count][mydepth-1]]])
    return np.array(x_batch,dtype='float32'), np.array(y_batch,dtype='float32')

def ncubeCreateBatch(batch_size, depth, orderNum, token='MLN'):
    if token is 'MLN':
        return ncubeCreateBatchMLN(batch_size, depth, orderNum)
    return ncubeCreateBatchSRNN(batch_size, depth, orderNum)

# Generates a balanced scrambled dataset
def generateBalancedScrambles(scramble_size, max_len, orderNum=2):
    scrambles=[]
    for curScramCount in range(scramble_size):
        temp =[]
        while len(temp) <= (curScramCount % max_len):
            temp.append(random.choice(list(actionAnti.keys())))
            temp = cleanUpScramble(temp)
        scrambles.append(temp[:])
    return scrambles


# Generate scrambles of fixed size
def generateFixedScrambles(scramble_size, max_len, orderNum=2):
    scrambles = []
    for curScramCount in range(scramble_size):
        temp = []
        while len(temp) < max_len:
            temp.append(random.choice(list(actionAnti.keys())))
            temp = cleanUpScramble(temp)
        scrambles.append(temp[:])
    return scrambles

# Generates random scramble sequences
def generateRandomScrambles(scramble_size, max_len, random_split=1.0, orderNum=2):
    scrambles = []
    for curScramCount in range(scramble_size):
        temp = []
        if curScramCount < (random_split*scramble_size):
            tempSize = random.choice(range(max_len))
        else:
            tempSize = max_len
        while len(temp) <= tempSize:
            temp.append(random.choice(list(actionAnti.keys())))
            temp = cleanUpScramble(temp)
        scrambles.append(temp[:])
    return scrambles

# The helper function
def generateScrambles(scramble_size, max_len, token='BALANCED', random_split=0.5, orderNum=2):
    if token is 'BALANCED':
        return generateBalancedScrambles(scramble_size, max_len, orderNum=orderNum )
    if token is 'RANDOM':
        return generateRandomScrambles(scramble_size, max_len, random_split, orderNum=orderNum)
    if token is 'FIXED':
        return generateFixedScrambles(scramble_size, max_len, orderNum=orderNum)
    else:
        print("INVALID SCRAMBLE TOKEN")
        return []


# This cleans up discrepancies in the scramble
# such as doing r and then r', it doesn't lead
# anywhere, we simply wish to avoid teaching
# that to the neural network
def cleanUpScramble(scramble, orderNum=2):
    i = 0
    while i < (len(scramble) - 1):
        if actionInverse[scramble[i]] == scramble[i+1]:
            del(scramble[i+1])
        else:
            i+=1
    if orderNum == 2:
        return cleanUpScrambleOrderTwo(scramble)
    return scramble

# This cleans up the anti actions I mentioned
# earlier. Take a look at the comment about
# it above (actionAnti Dictionary Defin.)
def cleanUpScrambleOrderTwo(scramble):
    i = 0
    while i < (len(scramble) - 1):
        if actionAnti[scramble[i]] == scramble[i+1]:
            del(scramble[i+1])
        else:
            i+=1
    return scramble