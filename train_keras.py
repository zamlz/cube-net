
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
training_epochs = 100
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
model.add(Dense(128, batch_input_shape=(batch_size, 1, 128),
					 activation='relu'))
model.add(Dense(128, batch_input_shape=(batch_size, 1, 128),
					 activation='relu'))
model.add(Dense(128, batch_input_shape=(batch_size, 1, 128),
					 activation='relu'))
model.add(Dense(128, batch_input_shape=(batch_size, 1, 128),
					 activation='relu'))
model.add(Dense(128, batch_input_shape=(batch_size, 1, 128),
					 activation='relu'))
model.add(Dense(128, batch_input_shape=(batch_size, 1, 12),
					 activation='relu'))
model.add(Dense(n_output, activation='relu'))
model.compile(optimizer='adam',
              loss='mean_squared_error',
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
		model.reset_states()
		ncube.displayCube(isColor=True)
		for _ in range(solv_limit):
			if ncube.isSolved():
				solv_count+=1
				print("Solved!")
				break
			vectorState = np.array([[ncube.constructVectorState(inBits=True)]]*batch_size, dtype='float32')
			# print(vectorState.shape)
			action = model.predict(vectorState, batch_size=batch_size, verbose=0)
			# print(action[0].shape)
			action = action[0]
			# print(action)
			action = np.argmax(action)
			# print(action)
			action = ct.indexToAction[action]
			ncube.minimalInterpreter(action)
			actionList.append(action)
		ncube.displayCube(isColor=True)
		print(scram)
		print(actionList)
		



# Start training
for epoch in range(training_epochs):

	print("\n"+("-"*20))
	print("EPOCH: "+str(epoch))
	for bat in range(training_batches):
		# Get the batch
		xb, yb = ct.ncubeCreateBatchLSTM(batch_size, depth, orderNum)		
		print("\nTrueBatch: "+str(bat+1)+"\n")
		for timestep in range(depth):
			
			xbi = xb[:,timestep:timestep+1,:]
			ybi = yb[:,timestep,:]
			
			#print(xb.shape)
			#print(xbi.shape)
			print("TimeStep: "+str(timestep+1))
			model.fit(xbi, ybi, nb_epoch=1, batch_size=batch_size, shuffle=False)
		model.reset_states()
	for i in range(10):
		print("TEST: "+str(i))
		testCube(1,'FIXED',20,1)
		model.reset_states()
		print("-"*20)