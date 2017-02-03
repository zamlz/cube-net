
import cube
import random

class Queue:
  "A container with a first-in-first-out (FIFO) queuing policy."
  def __init__(self):
    self.list = []
  
  def push(self,item):
    "Enqueue the 'item' into the queue"
    self.list.insert(0,item)

  def pop(self):
    """
      Dequeue the earliest enqueued item still in the queue. This
      operation removes the item from the queue.
    """
    return self.list.pop()

  def isEmpty(self):
    "Returns true if the queue is empty"
    return len(self.list) == 0


def bfs():
    ncube = cube.Cube(order=3)
    successorActions=['r','l','f','b','u','d']
    axisChange=['x','y','z']
    
    scramble= []
    for _ in range(500):
        scramble.append(random.choice(successorActions+axisChange))
    print("SCRAMBLED SEQUENCE:")
    print(scramble)
    for action in scramble:
        _, _ = ncube.minimalInterpreter(action, (False, 0))
    print("INITIAL CUBE STATE:")
    print(ncube.constructVectorState())
    ncube.displayCube(isColor=True)
    fringe = Queue()
    visited = []

    #mcube = cube.Cube(order=3)
    #mcube.displayCube(isColor=True)
    #vec= ncube.constructVectorState()
    #mcube.destructVectorState(vec)
    #mcube.displayCube(isColor=True)    
    maxActionLen = 0
    
    fringe.push((tuple(ncube.constructVectorState()), []))
    while not fringe.isEmpty():
        
        curState, curActions = fringe.pop()
        #print(curActions)
        if maxActionLen < len(curActions):
            maxActionLen = len(curActions)
            print(maxActionLen)

        if curState not in visited:
            visited.append(curState)

        if ncube.isSolved():
            return curActions

        for newAction in successorActions:
            ncube.destructVectorState(list(curState))
            _,_ = ncube.minimalInterpreter(newAction,(False, 0))
            newState = tuple(ncube.constructVectorState())
            fringe.push((newState, curActions + [newAction]))

    return []




def agentChoice(choice):
    if choice == 0:
        bfs()

if __name__ == "__main__":
    agentChoice(0)



