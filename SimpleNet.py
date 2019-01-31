import numpy as np
import ast
import os

def readSave(fileDir): #loads previously saved training data
    try:
        save = open(fileDir,"r")
        saveData = ast.literal_eval(save.read())
        return saveData[0],saveData[1],saveData[2] #returns layers, weights, biases
    except FileNotFoundError: #in case a file is inputted that does not exist
        print("invalid save")
        return None

class SimpleNet:
    def __init__(self,layers,weights=None,biases=None):
        self._layerList = layers #the structure of the neural net
        self._weightsList = [0] #all the weights
        self._biasesList = [0] #all the biases
        self._valuesList = [0]*len(self._layerList) #records the activations during foward propagation
        self._derivs = [0]*len(self._layerList) #records the derivatives of the activations
        self._wgrads = [0]*len(self._layerList) #the gradients of all the weights
        self._bgrads = [0]*len(self._layerList) #the gradients of all the biases
        if weights == None or biases == None:
            for i in range(1,len(self._layerList)): #initializes weights and biases
                self._weightsList.append((np.random.rand(self._layerList[i],self._layerList[i-1])-1)/(np.sqrt(self._layerList[i-1])))
                self._biasesList.append((np.random.rand(self._layerList[i],1)-1)/(np.sqrt(self._layerList[i-1])))
        else:
            for i in range(1,len(self._layerList)): #allows for the use of previously trained nets
                self._weightsList.append(np.array(weights[i]).reshape(self._layerList[i],self._layerList[i-1]))
                self._biasesList.append(np.array(biases[i]).reshape(len(biases[i]),1))

    def computeOutput(self,inputList): #foward propagation/returns output activations
        self._valuesList = [0]*len(self._layerList) #resets all the activation
        self._derivs = [0]*len(self._layerList) #resets the derivatives of the activations
        self._valuesList[0] = np.array(inputList,dtype=np.float32).reshape(len(inputList),1) #sets 
        for i in range(1,len(self._layerList)): #traverses each layer with an activation
            self._valuesList[i] = 1.0/(1+np.exp(-(self._weightsList[i] @ self._valuesList[i-1] + self._biasesList[i]))) #saves the activation from that layer
            self._derivs[i] = self._valuesList[i] * (1-self._valuesList[i]) #saves the derivative of that activation
        return self._valuesList[len(self._layerList)-1].tolist() #returns output (activation of the final layer)

    def backprop(self,actualList,LR): #calculates gradient for a single input and adds it to previously calculated gradients
        delta = self._derivs[len(self._layerList)-1]*(self._valuesList[len(self._layerList)-1] - np.array(actualList).reshape(len(actualList),1)) #used to calculate gradient
        for i in range(len(self._layerList)-1,0,-1): #traverses each layer with weight(s)/bias(es) backwards
            self._wgrads[i] += LR * (delta @ self._valuesList[i-1].transpose()) #weight gradients of layer i
            self._bgrads[i] += LR * delta #bias gradients of layer i
            delta = self._derivs[i-1] * (self._weightsList[i].transpose() @ delta) #updates for the next layer
    
    def applyGradient(self): #applies and resets previously calculated gradients
        for i in range(1,len(self._layerList)): #traverses each layer with weight(s)/bias(es)
            self._weightsList[i] -= self._wgrads[i] #applies weight gradient
            self._biasesList[i] -= self._bgrads[i] #applies bias gradient
        self._wgrads = [0]*len(self._layerList) #resets weight gradients
        self._bgrads = [0]*len(self._layerList) #resets bias gradients

    def getTrainingData(self): #returns the weights and biases as pure lists
        weights = [0]
        biases = [0]
        for i in range(1,len(self._layerList)):
            weights.append(self._weightsList[i].tolist()) #transforms weightList to a list of lists
            biases.append(self._biasesList[i].tolist()) #transforms biasList to a list of list
        return weights, biases
    
    def writeSave(self,fileName): #saves training data in a way that if you stop mid-write it will not delete your training data
        save = open("tempSave.txt","w+") #would not recomend keeping a file called tempSave.txt
        weights = [0]
        biases = [0]
        for i in range(1,len(self._layerList)):
            weights.append(self._weightsList[i].tolist()) #transforms weightList to a list of lists
            biases.append(self._biasesList[i].tolist()) #transforms biasList to a list of list
        saveData = [] #combines them for easy read/write
        saveData.append(self._layerList)
        saveData.append(weights)
        saveData.append(biases)
        save.write(str(saveData)) #writes save data to tempSave.txt
        try: #deletes old save if it exists
            os.remove(fileName)
        except OSError:
            pass
        save.close()
        os.rename("tempSave.txt",fileName) #renames new save to file name specified 