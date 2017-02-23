
import cube
import random
import numpy as np 
import os 
import cubeTrain as ct 

from keras.models import Sequential
from keras.layers import LSTM, Dense


# Create the nxn cube
orderNum = 2
ncube = cube.Cube(order=orderNum)
depth = 6


# Define the training parameters
training_epochs = 20
training_batches = 100
batch_size = 5000

# Verification Parameters
display_step = 1
test_data_size = 1000

# Solving Test Parameters
total_solv_trails = 100
solvable_limit = 50
solvable_step = 10


# Build the Keras LSTM Model
n_input = len(ncube.constructVectorState(inBits=True))
n_output = 12
timestep = depth

model = Sequential()
model.add(LSTM(128, stateful=True, 
					batch_input_shape=(batch_size, 1, n_input)))
#model.add(LSTM(128, stateful=True, 
#					batch_input_shape=(batch_size, 1, 128)))
#model.add(LSTM(128, stateful=True, 
#					batch_input_shape=(batch_size, 1, 128)))
model.add(Dense(128, batch_input_shape=(batch_size, 1, 12),
					 activation='softmax'))
model.add(Dense(n_output, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Print a summary of the model
model.summary()


# Cube testing function
def testCube(test_size, token, solv_limit, display_step):
	solv_count = 0
	scrambles = ct.generateScrambles(scramble_size=test_size,
									 max_len=depth,
									 token=token,
									 orderNum=2)
	# go through each scramble.
	for scram in scrambles:
		ncube = cube.Cube(order=orderNum)
		for action in scram:
			ncube.minimalInterpreter(action)
		actionList = []
		ncube.displayCube(isColor=True)
		for _ in range(solv_limit):
			if ncube.isSolved():
				solv_count+=1
				break
			vectorState = np.array([[ncube.constructVectorState(inBits=True)]], dtype='float32')
			action = model.predict(vectorState, batch_size=1)
			print(action)


# Start training
for epoch in range(training_epochs):

	print("\n"+("-"*20))
	print("EPOCH: "+str(epoch))
	# Get the batch
	xb, yb = ct.ncubeCreateBatchLSTM(batch_size, depth, orderNum)		
	
	for timestep in range(depth):
		xbi = xb[:,timestep:timestep+1,:]
		ybi = yb[:,timestep,:]
		model.train_on_batch(xbi, ybi)
	model.reset_states()
testCube(1,'FIXED',20,1)