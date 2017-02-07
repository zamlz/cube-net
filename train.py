
# _             _                     
#| |_ _ __ __ _(_)_ __    _ __  _   _ 
#| __| '__/ _` | | '_ \  | '_ \| | | |
#| |_| | | (_| | | | | |_| |_) | |_| |
# \__|_|  \__,_|_|_| |_(_) .__/ \__, |
#                        |_|    |___/ 

# This is a Simple Feed-Forward Model.

import cube
import random
import tensorflow as tf
import numpy as np 
import os

# Possible Values: FNN, CNN, RNN
NETWORK_TYPE = 'FNN'

# Create a nxn Cube
orderNum = 3
ncube = cube.Cube(order=orderNum)

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
}

# Given an output vector, you can
# find the corresponding action
vectorToAction={
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

# A small collection of gods number
numGod ={
    2:7,
    3:6,
}


# Creates a batched dataset
def ncubeCreateBatch(batch_size):
    x_batch=[]
    y_batch=[]
    for cur_batch in range(batch_size):
        ncube = cube.Cube(order=orderNum)
        if cur_batch > (batch_size/3.0):
            scramble = generateRandomScramble(scramble_size=numGod[orderNum],allowRandomSize=False)
        else:        
            scramble = generateRandomScramble(scramble_size=numGod[orderNum])
        for action in scramble:
            ncube.minimalInterpreter(action)
        x_batch.append(ncube.constructVectorState(inBits=True))
        y_batch.append(actionVector[actionInverse[scramble[-1]]])

    return np.array(x_batch,dtype='float32'), np.array(y_batch,dtype='float32')


# Generates random scramble sequences
def generateRandomScramble(scramble_size, allowRandomSize=True):
    scramble = []
    if allowRandomSize:
        scramble_size = random.choice(range(scramble_size)) + 1
    else:
        scramble_size = scramble_size
    for _ in range(scramble_size):
        scramble.append(random.choice(list(actionVector.keys())))
    if len(scramble) > 1:
        scramble = cleanUpScramble(scramble)
    if scramble == []:
        scramble.append(random.choice(list(actionVector.keys())))
    return scramble


# This cleans up discrepancies in the scramble
# such as doing r and then r', it doesn't lead
# anywhere, we simply wish to avoid teaching
# that to the neural network
def cleanUpScramble(scramble):
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


# Define Network Topolgy
n_input = len(ncube.constructVectorState(inBits=True))
n_hidden_1 = 1024
n_hidden_2 = 512
n_hidden_3 = 256
n_output = 12     # There are only 12 possible actions.


# Create the input and output variables
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_output])
keepratio = tf.placeholder(tf.float32)
stddev = 0.05


#
#   FEED FORWARD STUFF TYPICAL NEURAL NETWORK
#
# network Parameters For 
if NETWORK_TYPE is 'FNN':
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], stddev=stddev)),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], stddev=stddev)),
        'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3], stddev=stddev)),
        'out': tf.Variable(tf.random_normal([n_hidden_3, n_output], stddev=stddev))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'b3': tf.Variable(tf.random_normal([n_hidden_3])),
        'out': tf.Variable(tf.random_normal([n_output]))
    }


# Create the network
def FFNN(_X, _weights, _biases, _keep_prob):
    x_1 = tf.nn.relu(tf.add(tf.matmul(_X, _weights['h1']), _biases['b1']))
    layer_1 = x_1
    x_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, _weights['h2']), _biases['b2']))
    layer_2 = x_2
    x_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, _weights['h3']), _biases['b3']))
    layer_3 = tf.nn.dropout(x_3, _keep_prob)
    return (tf.matmul(layer_3, _weights['out']) + _biases['out'])


#
#   CONVOLUTIONAL NEURAL NETWORK STFF
#
dimOrder = int(len(ncube.constructVectorState(inBits=True))**0.5)
numConvLayers = 2
cnv = dimOrder // (numConvLayers*2)

if NETWORK_TYPE is 'CNN':
    weights  = {
        'wc1': tf.Variable(tf.truncated_normal([3, 3, 1, 64], stddev=stddev)),
        'wc2': tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=stddev)),
        'wc3': tf.Variable(tf.truncated_normal([3, 3, 128, 256], stddev=stddev)),
        'wd1': tf.Variable(tf.truncated_normal([cnv*cnv*256, 1024], stddev=stddev)),
        'wd2': tf.Variable(tf.truncated_normal([1024, n_output], stddev=stddev))
    }
    biases   = {
        'bc1': tf.Variable(tf.random_normal([64], stddev=0.1)),
        'bc2': tf.Variable(tf.random_normal([128], stddev=0.1)),
        'bc3': tf.Variable(tf.random_normal([256], stddev=0.1)),
        'bd1': tf.Variable(tf.random_normal([1024], stddev=0.1)),
        'bd2': tf.Variable(tf.random_normal([n_output], stddev=0.1))
    }


def CONV(_input, _w, _b, _keepratio):
    # INPUT
    _input_r = tf.reshape(_input, shape=[-1, dimOrder, dimOrder, 1])
    # CONV LAYER 1
    _conv1 = tf.nn.conv2d(_input_r, _w['wc1'], strides=[1, 1, 1, 1], padding='SAME')
    _mean, _var = tf.nn.moments(_conv1, [0, 1, 2])
    _conv1 = tf.nn.batch_normalization(_conv1, _mean, _var, 0, 1, 0.0001)
    _conv1 = tf.nn.relu(tf.nn.bias_add(_conv1, _b['bc1']))
    _pool1 = tf.nn.max_pool(_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    _pool_dr1 = tf.nn.dropout(_pool1, _keepratio)
    # CONV LAYER 2
    _conv2 = tf.nn.conv2d(_pool_dr1, _w['wc2'], strides=[1, 1, 1, 1], padding='SAME')
    _mean, _var = tf.nn.moments(_conv2, [0, 1, 2])
    _conv2 = tf.nn.batch_normalization(_conv2, _mean, _var, 0, 1, 0.0001)
    _conv2 = tf.nn.relu(tf.nn.bias_add(_conv2, _b['bc2']))
    _pool2 = tf.nn.max_pool(_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    _pool_dr2 = tf.nn.dropout(_pool2, _keepratio)
    # CONV LAYER 3
    _conv3 = tf.nn.conv2d(_pool_dr2, _w['wc3'], strides=[1,1,1,1], padding='SAME')
    _mean, _var = tf.nn.moments(_conv3, [0,1,2])
    _conv3 = tf.nn.batch_normalization(_conv3, _mean, _var, 0, 1, 0.0001)
    _conv3 = tf.nn.relu(tf.nn.bias_add(_conv3, _b['bc3']))
    #_pool3 = tf.nn.max_pool(_conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    #_pool_dr3 = tf.nn.dropout(_pool3, _keepratio)
    # VECTORIZE
    _dense1 = tf.reshape(_conv3, [-1, _w['wd1'].get_shape().as_list()[0]])
    # FULLY CONNECTED LAYER 1
    _fc1 = tf.nn.relu(tf.add(tf.matmul(_dense1, _w['wd1']), _b['bd1']))
    print(_fc1.get_shape())
    _fc_dr1 = tf.nn.dropout(_fc1, _keepratio)
    # FULLY CONNECTED LAYER 2
    _out = tf.add(tf.matmul(_fc_dr1, _w['wd2']), _b['bd2'])
    # RETURN
    out = { 'input_r': _input_r, 'conv1': _conv1, 'pool1': _pool1, 'pool1_dr1': _pool_dr1,
        'conv2': _conv2, 'pool2': _pool2, 'pool_dr2': _pool_dr2, 'dense1': _dense1,
        'fc1': _fc1, 'fc_dr1': _fc_dr1, 'out': _out
    }
    return out['out']


# Lets party

# Model 
if NETWORK_TYPE is 'FNN':
    model = FFNN(x, weights, biases, keepratio)
elif NETWORK_TYPE is 'CNN':
    model = CONV(x, weights, biases, keepratio)

# Cost Type
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(model, y))
# Optimizer
optm = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
# Correcion
corr = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
# Accuracy
accr = tf.reduce_mean(tf.cast(corr, "float"))
# Prediction
pred = tf.argmax(model, 1)


# Initialize everything up to this point
init = tf.initialize_all_variables()
#init = tf.global_variables_initializer()
print("CUBENET NEURAL NETWORK (",NETWORK_TYPE,") IS READY. ")


# Define the training parameters
training_epochs = 20
training_batches = 100
batch_size = 100
# Verification Paramters
display_step = 1
test_data_size = 1000
# Solving Paramters
total_solv_trials = 100
solvable_limit = 50
solvable_step = 5


# Create the Saver Object and directory to save in
saver = tf.train.Saver()
ckpt_dir = "./ckpt_dir"
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)


# Launch the tensorflow session
sess = tf.Session()
sess.run(init)


# Start the training
print("\nTRAINING HAS BEGUN...\n")
for epoch in range(training_epochs):
    avg_cost = 0.0

    # Each Epoch goes through a large set of batches
    # The exact values are defined above
    # Each Batch is a unique randomly generated sequence
    # from the rubiks cube
    for i in range(training_batches):
        #print(i)
        batch_x, batch_y = ncubeCreateBatch(batch_size)
        if NETWORK_TYPE is 'FNN':
            dictTemp = {x: batch_x, y: batch_y, keepratio: 0.6}
        elif NETWORK_TYPE is 'CNN':
            dictTemp = {x:batch_x, y:batch_y, keepratio: 0.7}
        sess.run(optm, feed_dict=dictTemp)
        dictTemp = {x: batch_x, y: batch_y, keepratio: 1.0}
        avg_cost+=sess.run(cost,feed_dict=dictTemp)
    avg_cost = avg_cost / training_batches

    # Save the model on every epoch
    save_path = saver.save(sess, ckpt_dir+"/model.ckpt")
    print("Model saved in File : %s" % save_path)
    
    # Display details of the epoch at certain intervals
    if (epoch + 1) % display_step == 0:
        
        # Epoch Stats
        print("\n----------------------------------------------------------------")
        print("Epoch: %03d/%03d cost: %.9f" % (epoch+1, training_epochs, avg_cost))
        
        # Test Data Stats
        test_x, test_y = ncubeCreateBatch(test_data_size)
        dictTemp = {x: test_x, y: test_y, keepratio: 1.0}
        test_acc = sess.run(accr, feed_dict=dictTemp)
        print("Test Accuracy: %.3f" % (test_acc))
        
        # Solving stats:
        solv_count = 0  # the amount of correct solves
        for solv_index in range(total_solv_trials): 
            ncube = cube.Cube(order=orderNum)
            
            # We must generate Larger scrambles here to emulate
            # a real world scramble
            if solv_index > (total_solv_trials/3.0):
                scramble = generateRandomScramble(scramble_size=numGod[orderNum],
                                                  allowRandomSize=False)
            else:
                scramble = generateRandomScramble(scramble_size=numGod[orderNum])
            for action in scramble:
                ncube.minimalInterpreter(action)
            
            # Display the cetain scrambled cubes
            if (solv_index+1) % solvable_step == 0:
                print("Trial: ", solv_index+1)
                print("Scramble: ", scramble)
                ncube.displayCube(isColor=True)
            
            # Time to test our network
            actionList = []
            for i in range(solvable_limit):
                # If we have solved the puzzle
                if ncube.isSolved():
                    solv_count+=1
                    break
                # Otherwise, lets apply the predicition of our network
                # to the model
                #----------
                # Get the state of the cube
                vectorState = []
                vectorState.append(ncube.constructVectorState(inBits=True))
                cubeState = np.array(vectorState, dtype='float32')
                # Apply the model
                dictTemp = {x:cubeState, keepratio:1.0}
                result = sess.run(pred, feed_dict=dictTemp)
                # Apply the result to the cube and save it
                actionList.append(vectorToAction[list(result)[0]])
                ncube.minimalInterpreter(actionList[-1])

            # Display certain end states of the cube
            if (solv_index+1) % solvable_step == 0:
                print("ActionList: ", actionList)
                ncube.displayCube(isColor=True)

        # What were the solve results?
        print("Practical Test: %03d/%03d -> %.3f" % (solv_count, total_solv_trials, solv_count/(total_solv_trials*1.0)))
    else:
        print("\nEPOCH: %03d" % (epoch+1))

# Training has been completed
print("Optimization Done!")
