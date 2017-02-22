
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
import cubeTrain as ct

# Possible Values: RNN
NETWORK_TYPE = 'SRNN'
DEPTH = 6

# Create a nxn Cube
orderNum = 2
ncube = cube.Cube(order=orderNum)


# Define the training parameters
training_epochs = 20
training_batches = 100
batch_size = 5000

# Verification Paramters
display_step = 1
test_data_size = 1000

# Solving Paramters
total_solv_trials = 100
solvable_limit = 50
solvable_step = 10

# Define Network Topolgy
n_input = len(ncube.constructVectorState(inBits=True))
n_steps = DEPTH
n_output = 12     # There are only 12 possible actions.
if NETWORK_TYPE is 'SRNN':
    n_input += n_output

# Define the layers of the MLN
mln_layers = 8
mln_info =[n_input] + [128]*mln_layers + [n_output]


# Create the input and output variables
if NETWORK_TYPE is 'SRNN':
    x = tf.placeholder("float", [None, n_input])
elif NETWORK_TYPE is 'LSTM':
    x = tf.placeholder("float", [None, n_steps, n_input])
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

def generateLSTM()

# Lets party

# Model 
if NETWORK_TYPE is 'SRNN':
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


# Initialize everything up to this point
init = tf.initialize_all_variables()
#init = tf.global_variables_initializer()
print("CUBENET NEURAL NETWORK (",NETWORK_TYPE,") IS READY. ")


# Create the Saver Object and directory to save in
saver = tf.train.Saver()
ckpt_dir = "./ckpt_dir"
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)


# Launch the tensorflow session
sess = tf.Session()
sess.run(init)

# Save the logs
summary_writer = tf.train.SummaryWriter('./ckpt_dir/logs', graph=sess.graph)

def testCube(test_size, token, solv_limit, display_step):
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
        lastMove = [0,0,0,0,0,0,0,0,0,0,0,0]
        for _ in range(solv_limit):
            if ncube.isSolved():
                solv_count+=1
                break
            vectorState = []
            vectorState.append(ncube.constructVectorState(inBits=True)+lastMove)
            cubeState = np.array(vectorState, dtype='float32')
            # Apply the model
            dictTemp = {x:cubeState, keepratio:1.0}
            result = sess.run(pred, feed_dict=dictTemp)
            # Apply the result to the cube and save it
            actionList.append(ct.indexToAction[list(result)[0]])
            lastMove = ct.indexToVector[list(result)[0]]
            ncube.minimalInterpreter(actionList[-1])
        if (scrIndex+1) % display_step == 0:
            ncube.displayCube(isColor=True)
            print("SCRAMBLE: ", scrambles[scrIndex])
            print("ACTION: ", actionList)
    print("Test Results (%s): %03d/%03d -> %.3f" % 
         (token, solv_count, test_size, solv_count/(test_size*1.0)))



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
        batch_x, batch_y = ct.ncubeCreateBatch(batch_size, DEPTH,orderNum, NETWORK_TYPE)
        dictTemp = {x: batch_x, y: batch_y, keepratio: 0.6}
        sess.run(optm, feed_dict=dictTemp)
        dictTemp = {x: batch_x, y: batch_y, keepratio: 1.0}
        avg_cost+=sess.run(cost,feed_dict=dictTemp)
    avg_cost = avg_cost / training_batches
    
    # Display details of the epoch at certain intervals
    if (epoch + 1) % display_step == 0:
        
        # Epoch Stats
        print("\n----------------------------------------------------------------")
        print("Epoch: %03d/%03d cost: %.9f" % (epoch+1, training_epochs, avg_cost))
        
        # Test Data Stats
        test_x, test_y = ct.ncubeCreateBatch(test_data_size, DEPTH, orderNum, NETWORK_TYPE)
        dictTemp = {x: test_x, y: test_y, keepratio: 1.0}
        test_acc = sess.run(accr, feed_dict=dictTemp)
        print("Test Accuracy: %.3f" % (test_acc))
        
        # Solving Stats
        testCube(total_solv_trials,'BALANCED', solvable_limit, solvable_step)
        #testCube(total_solv_trials,'RANDOM', solvable_limit, solvable_step)
        testCube(total_solv_trials,'FIXED', solvable_limit, solvable_step)

        # Save the model on every display stepped epoch
        save_path = saver.save(sess, ckpt_dir+"/model.ckpt")
        print("Model saved in File : %s" % save_path)

testCube(1000,'BALANCED', solvable_limit, solvable_step)
#testCube(1000,'RANDOM', solvable_limit, solvable_step)
testCube(1000,'FIXED', solvable_limit, solvable_step)


# Training has been completed
print("Optimization Done!")
