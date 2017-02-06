
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

# Create a nxn Cube
orderNum = 2
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
    2:11,
    3:20,
}


# Creates a batched dataset
def ncubeCreateBatch(batch_size):
    x_batch=[]
    y_batch=[]
    for _ in range(batch_size):
        ncube = cube.Cube(order=orderNum)
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
n_hidden_1 = n_input
n_hidden_2 = 48
n_hidden_3 = 24
n_output = 12     # There are only 12 possible actions.


# Create the input and output variables
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_output])
dropout_keep_prob = tf.placeholder("float")


# network Parameters
stddev = 0.05
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


# Lets party
model = FFNN(x, weights, biases, dropout_keep_prob)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(model, y))
optm = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
corr = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
accr = tf.reduce_mean(tf.cast(corr, "float"))
pred = tf.argmax(model, 1)


# Initialize everything up to this point
init = tf.initialize_all_variables()
#init = tf.global_variables_initializer()
print("CUBENET FEED FORWARD NEURAL NETWORK IS READY.")


# Define the training parameters
training_epochs = 500
training_batches = 100
batch_size = 500
# Verification Paramters
display_step = 1
test_data_size = 1000
# Solving Paramters
total_solv_trials = 250
solvable_limit = 200
solvable_step = 50


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
        sess.run(optm, feed_dict={x: batch_x, y: batch_y, dropout_keep_prob: 0.6})
        avg_cost+=sess.run(cost,feed_dict={x:batch_x, y:batch_y, dropout_keep_prob: 1.0})
    avg_cost = avg_cost / training_batches

    
    
    # Display details of the epoch at certain intervals
    if (epoch + 1) % display_step == 0:
        
        # Epoch Stats
        print("\n----------------------------------------------------------------")
        print("Epoch: %03d/%03d cost: %.9f" % (epoch+1, training_epochs, avg_cost))
        
        # Test Data Stats
        test_x, test_y = ncubeCreateBatch(test_data_size)
        test_acc = sess.run(accr, feed_dict={x:test_x,y:test_y,dropout_keep_prob:1.0})
        print("Test Accuracy: %.3f" % (test_acc))
        
        # Solving stats:
        solv_count = 0  # the amount of correct solves
        for solv_index in range(total_solv_trials): 
            ncube = cube.Cube(order=orderNum)
            
            # We must generate Larger scrambles here to emulate
            # a real world scramble
            scramble = generateRandomScramble(scramble_size=numGod[orderNum],
                                              allowRandomSize=False)
            for action in scramble:
                ncube.minimalInterpreter(action)
            
            # Display the cetain scrambled cubes
            if (solv_index+1) % solvable_step == 0:
                print("Trial: ", solv_index)
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
                result = sess.run(pred, feed_dict={x:cubeState, dropout_keep_prob:1.0})
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

# Save the model
save_path = saver.save(sess, ckpt_dir+"/model.ckpt")
print("Model saved in File : %s" % save_path)


# Training has been completed
print("Optimization Done!")
