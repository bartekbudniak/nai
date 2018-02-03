import random
import math
import numpy
from PIL import Image

class Network:

    def __init__(self, a):
        self.numberOfTests = 21
        self.size_1 = 100
        self.size_2 = 10
        self.size_3 = 2
        self.bias_2 = [0.6 for x in range(self.size_2)]
        self.bias_3 = [0.6 for x in range(self.size_3)]
        self.x_1 = [0.0 for x in range(self.size_1)]
        self.x_2 = [0.0 for x in range(self.size_2)]
        self.x_3 = [0.0 for x in range(self.size_3)]
        self.weight_1 = [[random.random()/10.0 for x in range(self.size_1)] for y in range(self.size_2)]
        self.weight_2 = [[random.random()/10.0 for x in range(self.size_2)] for y in range(self.size_3)]
        if a != 0:
            file1 = open('w1.txt', 'r').read()
            lines1 = file1.split('\n')
            i = 0
            for row in range(0, self.size_2):
                for column in range(0, self.size_1):
                    self.weight_1[row][column] = float(lines1[i])
                    i = i + 1
            #file1.close()

            file2 = open('w2.txt', 'r').read()
            lines2 = file2.split('\n')
            i = 0
            for row in range(0, self.size_3):
                for column in range(0, self.size_2):
                    self.weight_2[row][column] = float(lines2[i])
                    i = i + 1
            #file2.close()

    def downloadData(self, numberOfSample):
        nameOfImage = str(numberOfSample) + ".bmp"
        image = Image.open(nameOfImage, "r")
        pix_val = list(image.getdata())
        #print(pix_val)
        #pix_val_flat = [x for sets in pix_val for x in sets]
        for i in range(0, self.size_1):
            self.x_1[i] = float(pix_val[i][0])/255.0
            #print(self.x_1[i])

    def sigmoidFunction_2(self):
        for numberOfNeuron in range(0, self.size_2):
            sum = 0.0
            for i in range(0, self.size_1):
                sum = sum + self.weight_1[numberOfNeuron][i] * self.x_1[i]
                #print(sum)
            sum = sum + self.bias_2[numberOfNeuron]
            #print(numpy.exp(-1*sum))
            self.x_2[numberOfNeuron] = 1.0 / (1.0 + numpy.exp(-1*sum))
            #print(self.x_2[numberOfNeuron] )

    def sigmoidFunction_3(self):
        for numberOfNeuron in range(0, self.size_3):
            sum = 0.0
            for i in range(0, self.size_2):
                sum += self.weight_2[numberOfNeuron][i] * self.x_2[i]
            sum += self.bias_3[numberOfNeuron]
            self.x_3[numberOfNeuron] = 1.0 / (1.0 + math.exp(-1*sum))
        self.normalizeVector()

    def normalizeVector(self):
        sum = 0.0
        for numberOfNeuron in range(0, self.size_3):
            sum += self.x_3[numberOfNeuron]
        for numberOfNeuron in range(0, self.size_3):
            self.x_3[numberOfNeuron] /= sum

    def writeResultInFile(self, numberOfSample):
        file = open("results.txt", "a")
        file.write('Sample number {0}: {1:4f} vs. {2:4f}\n'.format(numberOfSample, self.x_3[0], self.x_3[1])) #zmienić przy zmianie liczby wyjść
        file.close()

    def saveWeightsInFile(self):
        file1 = open("w1.txt", "w")
        for i in range(0, self.size_2):
            for j in range(0, self.size_1):
                file1.write("{}\n".format(self.weight_1[i][j]))
        file1.close()

        file2 = open("w2.txt", "w")
        for i in range(0, self.size_3):
            for j in range(0, self.size_2):
                file2.write("{}\n".format(self.weight_2[i][j]))
        file2.close()

    def learn(self):
        file = open("results.txt", "w")
        file.close()
        for i in range(1, 10):
            for numberOfSample in range(0, self.numberOfTests):
                self.downloadData(numberOfSample)
                self.sigmoidFunction_2()
                self.sigmoidFunction_3()
                previousTotalError = self.calcTotalError(numberOfSample)
                for row in range(0, self.size_2):
                    for column in range(0, self.size_1):
                        temp = self.weight_1[row][column]
                        self.weight_1[row][column] = random.random() / 10.0
                        self.sigmoidFunction_2()
                        self.sigmoidFunction_3()
                        if self.calcTotalError(numberOfSample) > previousTotalError:
                            self.weight_1[row][column] = temp
                self.writeResultInFile(numberOfSample)

            for numberOfSample in range(0, self.numberOfTests):
                self.downloadData(numberOfSample)
                self.sigmoidFunction_2()
                self.sigmoidFunction_3()
                previousTotalError = self.calcTotalError(numberOfSample)
                for row in range(0, self.size_3):
                    for column in range(0, self.size_2):
                        temp = self.weight_2[row][column]
                        self.weight_2[row][column] = random.random() / 10.0
                        self.sigmoidFunction_2()
                        self.sigmoidFunction_3()
                        if self.calcTotalError(numberOfSample) > previousTotalError:
                            self.weight_2[row][column] = temp
                self.writeResultInFile(numberOfSample)

        self.saveWeightsInFile()

        '''for i in range(1, 10):
            for row in range(0, self.size_2):
                for column in range(0, self.size_1):
                    errorForAllData = 0.0
                    for numberOfSample in range(0, self.numberOfTests):
                        self.downloadData(numberOfSample)
                        self.sigmoidFunction_2()
                        self.sigmoidFunction_3()
                        errorForAllData += self.calcTotalError(numberOfSample)

                    temp = self.weight_1[row][column]
                    self.weight_1[row][column] = random.random() / 10.0

                    newErrorForAllData = 0.0
                    for numberOfSample in range(0, self.numberOfTests):
                        self.downloadData(numberOfSample)
                        self.sigmoidFunction_2()
                        self.sigmoidFunction_3()
                        newErrorForAllData += self.calcTotalError(numberOfSample)

                    if newErrorForAllData > errorForAllData:
                        self.weight_1[row][column] = temp

            for row in range(0, self.size_3):
                for column in range(0, self.size_2):
                    errorForAllData = 0.0
                    for numberOfSample in range(0, self.numberOfTests):
                        self.downloadData(numberOfSample)
                        self.sigmoidFunction_2()
                        self.sigmoidFunction_3()
                        errorForAllData += self.calcTotalError(numberOfSample)

                    temp = self.weight_2[row][column]
                    self.weight_2[row][column] = random.random() / 10.0

                    newErrorForAllData = 0.0
                    for numberOfSample in range(0, self.numberOfTests):
                        self.downloadData(numberOfSample)
                        self.sigmoidFunction_2()
                        self.sigmoidFunction_3()
                        newErrorForAllData += self.calcTotalError(numberOfSample)

                    if newErrorForAllData > errorForAllData:
                        self.weight_2[row][column] = temp
                for row in range(0, self.size_3):
                    for column in range(0, self.size_2):
                        previousTotalError = self.calcTotalError(numberOfSample)
                        temp = self.weight_2[row][column]
                        self.weight_2[row][column] = random.random() / 10.0
                        self.sigmoidFunction_2()
                        self.sigmoidFunction_3()
                        if self.calcTotalError(numberOfSample) > previousTotalError:
                            self.weight_2[row][column] = temp'''


    def calcTotalError(self, numberOfSample):
        totalError = 0.0
        #file = open("learningData.txt", "r")
        if numberOfSample % 2 == 0:
            output = [1, 0]                                 #wpisane na sztywno:(
        else:
            output = [0, 1]
        #file.close()
        for i in range(0, self.size_3):
            totalError += 0.5*(self.x_3[i] - output[i])**2
        return totalError

    #def backPropagation(self):

    def feedForward(self):
        file = open("results.txt", "a")
        file.write("************************************************\n")
        file.close()
        for i in range(1,6):
            nameOfTest = "test" + str(i)
            self.downloadData(nameOfTest)
            self.sigmoidFunction_2()
            self.sigmoidFunction_3()
            self.writeResultInFile(nameOfTest)



network = Network(0)
network.learn()
'''print(network.size_1)
print(network.size_2)
print(network.size_3)
print(network.bias_2)
print(network.bias_3)
print(network.weight_1)
print(network.weight_2)
print(network.x_1)
print(network.x_2)
print(network.x_3)'''
network2 = Network(1)
network2.feedForward()





