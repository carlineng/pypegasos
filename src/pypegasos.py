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
                    self.processBatch(currentBatch)
                    currentBatch = []

            self.processBatch(currentBatch)
            inputfile.close()

    def processBatch(self, currentBatch):
        k = len(currentBatch)
        updateSet = []
        # Determine which items in the batch contribute to loss
        for obs in currentBatch:
            y = int(obs[0])
            features = obs[0:]
            inner_product = self.vector.innerProduct(features)
            if y * inner_product < 1:
                updateSet.append(obs)
        
        eta = 1./(self.lamb * self.t)
        # TODO update weight vector with information from updateSet
