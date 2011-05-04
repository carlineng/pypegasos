#!/usr/bin/env python

from VowpalVector import *

class Pegasos:

    def __init__(self, k=5, nbits=18, lamb=.1):
        self.vector = VowpalVector(nbits)
        self.batchSize = k # Batch size. k=1 is SGD, k=N is batch GD
        self.t = 1
        self.lamb = lamb
        self.inputfile = None
        self.outputfile = None
    
    def setInputFile(self, filename):
        self.inputfile = filename

    def setOutputFile(self, filename):
        self.outputfile = filename

    def iterate(self, filename=None):
        """ Single pass over the input dataset"""
        if self.inputfile is None and filename is None:
            print "No input file specified."
        else:
            if filename is None:
                filename = self.inputfile

            currentBatch = []

            inputfile = open(filename, 'r')
            for line in inputfile:
                splitLine = line.split(',')
                currentBatch.append(splitLine)

                if len(currentBatch) >= self.batchSize:
                    self.vector.processBatch(currentBatch, self.t)
                    self.t = self.t + 1
                    currentBatch = []

            if len(currentBatch) > 0:
                self.vector.processBatch(currentBatch, self.t)

            inputfile.close()

def predict(self, instance):
    prediction = self.vector.innerProduct(instance)
    return prediction

def writeWeights(self, filename=None):
    if self.outputfile is None and filename is None:
        print "No input file specified."
    else:
        if filename is None:
            filename = self.outputfile

        outf = open(filename, 'w')
        for w in self.vector.weights:
        outf.write(str(w) + '\n')
        outf.close()

