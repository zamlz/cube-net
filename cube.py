
#            _                        
#  ___ _   _| |__   ___   _ __  _   _ 
# / __| | | | '_ \ / _ \ | '_ \| | | |
#| (__| |_| | |_) |  __/_| |_) | |_| |
# \___|\__,_|_.__/ \___(_) .__/ \__, |
#                        |_|    |___/ 

#
#   A rubiks cube program and library.
#   Running this program with the command
#   python cube.py will let you play
#   with the rubiks cube. 
#   Commands will be displayed as well.
#
#       - Amlesh Sivanantham
#

import sys
import os
from math import ceil
from termcolor import colored


class Cube:    

    tileChar     ='   '
    tileCharReal =' + '
    colorDict = {
        ' W ' : 'white',
        ' G ' : 'green',
        ' B ' : 'blue',
        ' R ' : 'red',
        ' O ' : 'magenta',
        ' Y ' : 'yellow',
    }
    faceDict = {
        'f' : 'Front',
        'r' : 'Right',
        'l' : 'Left',
        'u' : 'Up',
        'd' : 'Down',
        'b' : 'Back',
    }
    tileDict = {
        ' R ' : [0,0,1],
        ' O ' : [0,1,0],
        ' Y ' : [0,1,1],
        ' G ' : [1,0,0],
        ' B ' : [1,0,1],
        ' W ' : [1,1,0],
    }

    def __init__(self, order):
        self.order  = order
        self.front  = [[ ' W ' for y in range(order)] for x in range(order)]
        self.up     = [[ ' G ' for y in range(order)] for x in range(order)]
        self.down   = [[ ' B ' for y in range(order)] for x in range(order)]
        self.left   = [[ ' R ' for y in range(order)] for x in range(order)]
        self.right  = [[ ' O ' for y in range(order)] for x in range(order)]
        self.back   = [[ ' Y ' for y in range(order)] for x in range(order)]

    """
    Displays the ASCII Cube or the Colorized Cube
    """
    def displayCube(self, isColor=False):
        # Display the Top Portion
        for i in range(self.order):
            for j in range(self.order):
                sys.stdout.write(self.tileChar)
            for tile in self.up[i]:
                sys.stdout.write(self.getTileColor(tile[:],isColor))
            for j in range(self.order*2):
                sys.stdout.write(self.tileChar)
            sys.stdout.write('\n')
        # Display the middle section
        for i in range(self.order):
            for tile in self.left[i]:
                sys.stdout.write(self.getTileColor(tile[:],isColor))
            for tile in self.front[i]:
                sys.stdout.write(self.getTileColor(tile[:],isColor))
            for tile in self.right[i]:
                sys.stdout.write(self.getTileColor(tile[:],isColor))
            for tile in self.back[i]:
                sys.stdout.write(self.getTileColor(tile[:],isColor))
            sys.stdout.write('\n')
        # Display the Bottom Section
        for i in range(self.order):
            for j in range(self.order):
                sys.stdout.write(self.tileChar)
            for tile in self.down[i]:
                sys.stdout.write(self.getTileColor(tile[:],isColor))
            for j in range(self.order*2):
                sys.stdout.write(self.tileChar)
            sys.stdout.write('\n')
        sys.stdout.write('\n')

    def getTileColor(self, tile, isColor=False):
        if isColor:
            tile = colored(Cube.tileCharReal, Cube.colorDict[tile],
                           attrs=['reverse','blink'])
        return tile

    # takes a face character and rotates it in a given direction
    def __rotateFace(self,face='Front',dir='ClkWise',iter=1):
        # Choose the right face for the job
        if   face is 'Front' : tempFace = self.front
        elif face is 'Up'    : tempFace = self.up
        elif face is 'Down'  : tempFace = self.down
        elif face is 'Left'  : tempFace = self.left
        elif face is 'Right' : tempFace = self.right
        elif face is 'Back'  : tempFace = self.back
        else : print("ERROR")

        # Here is where the rotation algorithm happens
        # I could have made a simple one, but this was
        # much cooler to implement.
        N = self.order - 1
        if   dir is 'ClkWise'    :
            transformForward = lambda x,y: (-y, x)
            transformUpdate  = lambda n,s: (n - 2*s, 0)
            transformSkew    = lambda x,y: (x-1, y-1)
        elif dir is 'CntrClkWise':
            transformForward = lambda x,y: (y, -x)
            transformUpdate  = lambda n,s: (0, n - 2*s)
            transformSkew    = lambda x,y: (x+1, y-1)
        else: print("ERROR")

        for _ in range(iter):
            for i in range(ceil(self.order/2)):
                tfvec = transformUpdate(N,i)

                for j in range(i, N - i):
                    fi = i
                    tempTile = tempFace[fi][j]
                    for _ in range(3):
                        tempFace[fi][j] = tempFace[fi + tfvec[0]][j + tfvec[1]]
                        fi += tfvec[0]
                        j  += tfvec[1]    
                        tfvec = transformForward(tfvec[0],tfvec[1])
                    tempFace[fi][j] = tempTile
                    tfvec = transformForward(tfvec[0],tfvec[1])
                    tfvec = transformSkew(tfvec[0],tfvec[1])


    def rotateAlongAxis(self, axis, inverse=False):
        forward='ClkWise'
        backward='CntrClkWise'
        if inverse:
            forward='CntrClkWise'
            backward='ClkWise'

        if axis is 'z':
            self.__rotateFace(face='Front', dir=forward)
            self.__rotateFace(face='Back', dir=backward)
            self.__rotateFace(face='Up',    dir=forward)
            self.__rotateFace(face='Down',  dir=forward)
            self.__rotateFace(face='Left',  dir=forward)
            self.__rotateFace(face='Right', dir=forward)
            if not inverse:
                tempFace   = self.right[:]
                self.right =    self.up[:]
                self.up    =  self.left[:]
                self.left  =  self.down[:]
                self.down  =   tempFace[:]
            else:
                tempFace = self.right[:]
                self.right = self.down[:]
                self.down = self.left[:]
                self.left = self.up[:]
                self.up = tempFace[:]

        elif axis is 'y':
            self.__rotateFace(face='Up',   dir=forward)
            self.__rotateFace(face='Down',dir=backward)
            if not inverse:
                tempFace   =  self.left[:]
                self.left  = self.front[:]
                self.front = self.right[:]
                self.right =  self.back[:]
                self.back  =   tempFace[:]
            else:
                tempFace   =  self.back[:]
                self.back  = self.right[:]
                self.right = self.front[:]
                self.front =  self.left[:]
                self.left  =   tempFace[:]

        elif axis is 'x':
            self.__rotateFace(face='Back',  dir='ClkWise', iter=2)
            self.__rotateFace(face='Left',  dir=backward, iter=1)
            self.__rotateFace(face='Right',  dir=forward, iter=1)
            if not inverse:
                self.__rotateFace(face='Up', dir='ClkWise', iter=2)
                tempFace   =  self.back[:]
                self.back  =    self.up[:]
                self.up    = self.front[:]
                self.front =  self.down[:]
                self.down  =   tempFace[:]
            else:
                self.__rotateFace(face='Down', dir='ClkWise', iter=2)
                tempFace   =  self.back[:]
                self.back  =  self.down[:]
                self.down  = self.front[:]
                self.front =    self.up[:]
                self.up    =   tempFace[:]

        else:
            print("ERROR")

    def __resolveLayersOnlyFront(self, layer, inverse=False):
        N = self.order - 1
        # Your X Y coordinate naming system is shit
        # Fix this ASAP
        if not inverse    :
            transformForward = lambda x,y: (-y, x)
            transformUpdate  = lambda n,j: (n-j-layer, -j+layer)
            transformSkew    = lambda x,y: (x-1, y-1)
            tempFace_1       = self.down
            tempFace_2       = self.right
            tempFace_3       = self.up
            tempFace_4       = self.left
        else:
            transformForward = lambda x,y: (y, -x)
            transformUpdate  = lambda n,s: (j, n-j)
            transformSkew    = lambda x,y: (x+1, y-1)
            tempFace_1       = self.down
            tempFace_2       = self.left
            tempFace_3       = self.up
            tempFace_4       = self.right

        for j in range(self.order):
            i = layer
            tfvec = transformUpdate(N,j)
            tempTile = tempFace_1[i][j]

            tempFace_1[i][j] = tempFace_2[i + tfvec[0]][j + tfvec[1]]
            i+=tfvec[0]
            j+=tfvec[1]
            tfvec = transformForward(tfvec[0], tfvec[1])
            tempFace_2[i][j] = tempFace_3[i + tfvec[0]][j + tfvec[1]]
            i+=tfvec[0]
            j+=tfvec[1]
            tfvec = transformForward(tfvec[0], tfvec[1])
            tempFace_3[i][j] = tempFace_4[i + tfvec[0]][j + tfvec[1]]
            i+=tfvec[0]
            j+=tfvec[1]
            tfvec = transformForward(tfvec[0], tfvec[1])
            tempFace_4[i][j] = tempTile
            tfvec = transformForward(tfvec[0], tfvec[1])

        

    def rotateFaceReal(self, face, layer, inverse=False):
        if layer >= self.order:
            print("Layer Value Out of Bounds")

        if layer == 0:
            self.__rotateFace(face=face,
                              dir=('ClkWise' if not inverse else 'CntrClkWise'))

        if face is "Front":
            self.__resolveLayersOnlyFront(layer, inverse)
        elif face is "Back":
            self.rotateAlongAxis(axis='y')
            self.rotateAlongAxis(axis='y')
            self.__resolveLayersOnlyFront(layer, inverse)
            self.rotateAlongAxis(axis='y')
            self.rotateAlongAxis(axis='y')
        elif face is "Right":
            self.rotateAlongAxis(axis='y')
            self.__resolveLayersOnlyFront(layer, inverse)
            self.rotateAlongAxis(axis='y', inverse=True)
        elif face is "Left":
            self.rotateAlongAxis(axis='y', inverse=True)
            self.__resolveLayersOnlyFront(layer, inverse)
            self.rotateAlongAxis(axis='y')
        elif face is "Up":
            self.rotateAlongAxis(axis='x', inverse=True)
            self.__resolveLayersOnlyFront(layer, inverse)
            self.rotateAlongAxis(axis='x')
        elif face is "Down":
            self.rotateAlongAxis(axis='x')
            self.__resolveLayersOnlyFront(layer, inverse)
            self.rotateAlongAxis(axis='x', inverse=True)


    def minimalInterpreter(self, cmdString):
        inv = False
        lay = 0
        for command in cmdString:
            if command is '.':
                inv = not inv
            elif command.isdigit():
                lay = min(max(int(command)-1,0),self.order - 1)
            elif command in ['x','y','z']:
                self.rotateAlongAxis(command, inv)
                inv = False
                lay = 0
            elif command in Cube.faceDict.keys():
                self.rotateFaceReal(face=Cube.faceDict[command],layer=lay,inverse=inv)
                inv = False
                lay = 0

    def client(self, isColor=False):
        while True:
            clearScreen()
            self.displayCube(isColor=isColor)
            userString = str(input("\n---> "))
            self.minimalInterpreter(userString)
            print(self.constructVectorState(inBits=True)) 

    def constructVectorState(self, inBits=False):
        vector = []
        tileDictOrdTwo = {}
        faces = [self.front, self.back, self.right, self.left, self.up, self.down]
        bitValue = 1
        for face in faces:
            for faceRow in face:
                for faceTile in faceRow:
                    if inBits:
                        # If order two, we use relative bit coloring
                        if self.order is 2:
                            if faceTile in list(tileDictOrdTwo.keys()):
                                vector.extend(tileDictOrdTwo[faceTile])
                            else:
                                temp = []
                                if 32 & bitValue: temp.append(1) 
                                else: temp.append(0)
                                if 16 & bitValue: temp.append(1) 
                                else: temp.append(0)
                                if 8 & bitValue: temp.append(1) 
                                else: temp.append(0)
                                if 4 & bitValue: temp.append(1) 
                                else: temp.append(0)
                                if 2 & bitValue: temp.append(1) 
                                else: temp.append(0)
                                if 1 & bitValue: temp.append(1) 
                                else: temp.append(0)
                                bitValue*=2
                                tileDictOrdTwo[faceTile] = temp
                                vector.extend(temp)
                        else:
                            vector.extend(self.tileDict[faceTile])
                    else:
                        vector.append(faceTile.split()[0])
        return vector

    def destructVectorState(self, tileVector, inBits=False):
        count= 0
        faces = [self.front, self.back, self.right, self.left, self.up, self.down]
        #for face in faces:
        for f in range(len(faces)):
            for i in range(self.order):
                for j in range(self.order):
                    faces[f][i][j] = " "+tileVector[f*(self.order**2) + i*self.order + j]+" "

    def isSolved(self):
        faces = [self.front, self.back, self.right, self.left, self.up, self.down]
        for face in faces:
            for i in range(self.order):
                for j in range(self.order):
                    if face[i][j] != face[0][0]:
                        return False
        return True



def clearScreen():
    if os.name == "nt":
        os.system('cls')
    else:
        os.system('clear')


def main():
    ord = 'a'
    while not ord.isdigit():
        clearScreen()
        ord = input("\nEnter the order of the cube: ")
    cn = Cube(order=int(ord))
    cn.client(isColor=True)
    


if __name__ == "__main__":
    main()