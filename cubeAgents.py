
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


# Make this into A*
def bfs():
    ncube = cube.Cube(order=3)
    successorActionsNorm=['r','l','f','b','u','d']
    successorActionsInv=['.r','.l','.f','.b','.u','.d']
    successorActions = successorActionsInv + successorActionsNorm
    scramble= []
    for _ in range(5):
        scramble.append(random.choice(successorActions))
    print("SCRAMBLED SEQUENCE:")
    print(scramble)
    for action in scramble:
        ncube.minimalInterpreter(action)
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

        for newAction in successorActions:
            ncube.destructVectorState(list(curState))
            ncube.minimalInterpreter(newAction)
            if ncube.isSolved():
                return curActions+ [newAction]
            newState = tuple(ncube.constructVectorState())
            fringe.push((newState, curActions + [newAction]))

    return []




def agentChoice(choice):
    if choice == 0:
        print(bfs())

if __name__ == "__main__":
    agentChoice(0)



