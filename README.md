# CubeNet

A test of different neural network models to try and solve a 2x2x2 Rubiks Cube using TensorFlow. You can see a couple models have been trained. These can be tested with test_mlp.py.

The basic network learned to solve a Rubiks Cube of scramble length 6 with an accuracy of 47%. By also training the network to learn the previous move made, it was able to improve the accuracy of the network 52%.

#### Seen below is the model in action. 
Here it is attempting to solve a 2x2x2 but takes an interesting route to do so. (test_mlp.py)

![alt text](https://github.com/zAMLz/CubeNet/blob/master/pics/cube.JPG?raw=true "Here is a picture lol")

Here it is trying to solve a 3x3x3. It however cannot solve it, and struggles to make a decison once it has come to this point.

![alt text](https://github.com/zAMLz/CubeNet/blob/master/pics/cube2.jpg?raw=true "Here is a picture lol")

What we can infer from this is that the network has learned to solve the cube by attempting to create larger structures. However the networks can only prioritize making these structures and thus when stuck in a situation as seen above it is unable to solve it further since it would have to break the structure that it has made up to this point. What the network actually does is break the structure, but on the next iteration, it realizes that it must redo that inverse of that move since it creates a larger structure. However at this point, it finds itself in a cycle, constantly doing the same two moves over and over again. This problem was fixed by a small degree when training a model that also took the last action taken as input as well. When a scenario like the above occured, it would start to try various different moves until it could start building new better structures. However, as mentioned above, it only improved the solving rate by 5%. This is because there are many scenarios where is will stick get stuck in a completely new cube state that is not fully solved and thus it will continue trying various things and it will not have solved the cube.
