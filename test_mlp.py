#!/bin/python

# Test MLP
# ONLY WORKS FOR MLP

print("Start MLP Test")

import cube
import random
import tensorflow as tf
import numpy as np 
import os
import cubeTrain as ct 
from copy import deepcopy


# Model path
mpath='./model_2_2_moves_6_MLN_8L/model.ckpt'

# Various parameters
DEPTH = 6
orderNum = 2

ncube = cube.Cube(order=orderNum)

# Define the layers of the MLN
n_input = len(ncube.constructVectorState(inBits=True))
n_output = 12
mln_layers = 8
mln_info =[n_input] + [128]*mln_layers + [n_output]

# Some display stuff for the command line
displayStep = 10
limit = 20
trials = 10


# Create the input and output variables
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_output])
keepratio = tf.placeholder(tf.float32)
stddev = 0.05

#
#   MULTILAYER NEURAL NETWORK STUFF
#
weights = {}
biases = {}

def initWeight(shape, stddev):
    return tf.Variable(tf.random_normal(shape,stddev=stddev))

def initBias(shape):
    return tf.Variable(tf.random_normal(shape))

def initLayer(x, w, b):
    lxw = tf.matmul(x, w)
    lb  = tf.add(lxw, b)
    lr  = tf.nn.relu(lb)
    return lr

def finalLayer(x, w, b, keep_prob):
    ld = tf.nn.dropout(x, keep_prob)
    lxw =tf.matmul(ld, w)
    lb = tf.add(lxw, b)
    return lb

def generateMLN(X, keep_prob, mlnInfo):
    for i in range(1,len(mlnInfo)):
        weights[i] = initWeight([mlnInfo[i-1], mlnInfo[i]],0.05)
        biases[i]  = initBias([mlnInfo[i]])
    layers = [X]
    i = 1
    for _ in range(1, len(mlnInfo)-1):
        layers.append(initLayer(layers[i-1], weights[i], biases[i]))
        i+=1
    layers.append(finalLayer(layers[i-1],weights[i], biases[i], keep_prob))
    return layers[-1]


# Define the model
model = generateMLN(x, keepratio, mln_info)
# Cost Type
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=y))
# Optimizer
optm = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
# Correcion
corr = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
# Accuracy
accr = tf.reduce_mean(tf.cast(corr, "float"))
# Prediction
pred = tf.argmax(model, 1)


# Init the variables
init = tf.initialize_all_variables()


# lets load out model
# But lets ensure the file exits
saver = tf.train.Saver()
if not os.path.isfile(mpath):
    print("Unable to find model path: ", mpath)
    exit(1)

# Launch the tensorflow session
sess = tf.Session()
sess.run(init)

# Actually load the model now
saver.restore(sess, mpath)
print("MLP Model has been sucessfully restored")

# The testing function for the cube
def testCube(test_size, token, solv_limit, display_step):
    print("---------------------------------------------")
    print(token)
    print("---------------------------------------------")
    solv_count = 0
    scrambles = ct.generateScrambles(scramble_size=test_size,
        max_len=DEPTH, token=token, orderNum=2)
    # Go through each scramble
    for scrIndex in range(test_size):
        ncube = cube.Cube(order=orderNum)
        # Actually scramble the cube
        for action in scrambles[scrIndex]:
            ncube.minimalInterpreter(action)
        actionList=[]
        if (scrIndex+1) % display_step == 0:
            ncube.displayCube(isColor=True)
        # Solving phase
        for _ in range(solv_limit):
            if ncube.isSolved():
                solv_count+=1
                break
            vectorState = []
            vectorState.append(ncube.constructVectorState(inBits=True))
            cubeState = np.array(vectorState, dtype='float32')
            # Apply the model
            dictTemp = {x:cubeState, keepratio:1.0}
            result = sess.run(pred, feed_dict=dictTemp)
            # Apply the result to the cube and save it
            actionList.append(ct.indexToAction[list(result)[0]])
            ncube.minimalInterpreter(actionList[-1])
        if (scrIndex+1) % display_step == 0:
            ncube.displayCube(isColor=True)
            print("SCRAMBLE: ", scrambles[scrIndex])
            print("ACTION: ", actionList)
    return ("Test Results (%s): %03d/%03d -> %.3f" % 
         (token, solv_count, test_size, solv_count/(test_size*1.0)))

def testCubeBFS(test_size, token, solv_limit, display_step):
    print("---------------------------------------------")
    print("BFS: ",token)
    print("---------------------------------------------")
    solv_count = 0
    scrambles = ct.generateScrambles(scramble_size=test_size, max_len=DEPTH, token=token, orderNum=2)
    # Go through each scramble
    for scrIndex in range(test_size):
        ncube = cube.Cube(order=orderNum)
        # Actually scramble the cube
        for action in scrambles[scrIndex]:
            ncube.minimalInterpreter(action)
        if (scrIndex+1) % display_step == 0:
            ncube.displayCube(isColor=True)
        # Solving phase
        fringe = []
        actionList=[]
        fringe.append((deepcopy(ncube), actionList[:]))
        while len(actionList) <= DEPTH + 3:
            tempCube, actionList = fringe.pop()
            if tempCube.isSolved():
                solv_count+=1
                break
            
            cubeState = np.array([tempCube.constructVectorState(inBits=True)], dtype='float32')
            # Apply the model
            dictTemp = {x:cubeState, keepratio:1.0}
            result = sess.run(model, feed_dict=dictTemp)
            result = result[0]
            # Lets find the top two indicies
            maxi = 0
            oldi = 0
            maxval = -999999
            for i in range(len(result)):
                if result[i] > result[maxi]:
                    maxi = i
            maxval = -999999
            if maxi == oldi:
                oldi+=1
            for i in range(len(result)):
                if result[i] > result[oldi] and result[i] < result[maxi]:
                    oldi = i
            # Apply the result to the cube and save it
            tempCubeNew = deepcopy(tempCube)
            actionList.append(ct.indexToAction[maxi])
            tempCube.minimalInterpreter(actionList[-1])
            fringe = [(deepcopy(tempCube), actionList[:])] + fringe
            
            actionList[-1] = ct.indexToAction[oldi]
            tempCubeNew.minimalInterpreter(actionList[-1])
            fringe = [(deepcopy(tempCubeNew), actionList[:])] + fringe

        if (scrIndex+1) % display_step == 0:
            tempCube.displayCube(isColor=True)
            print("SCRAMBLE: ", scrambles[scrIndex])
            print("ACTION: ", actionList)
    return ("Test Results BFS (%s): %03d/%03d -> %.3f" % 
         (token, solv_count, test_size, solv_count/(test_size*1.0)))

temp = ""
temp += testCube(trials,'BALANCED', limit, displayStep) + "\n"
temp += testCube(trials,'FIXED', limit, displayStep) + "\n\n"

temp += testCubeBFS(trials,'BALANCED', limit, displayStep) + "\n"
temp += testCubeBFS(trials,'FIXED', limit, displayStep) + "\n"

print(temp)
