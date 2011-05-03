#!/usr/bin/env python

from VowpalVector import *

class Pegasos:

    def __init__(self, k=5, nbits=18, lamb=.1):
        self.weights = VowpalVector(nbits, lamb)
        self.k = k # Batch size. k=1 is SGD, k=N is batch GD
        self.t = 1
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

                if len(currentBatch) >= self.k:
                    self.processBatch(currentBatch)
                    currentBatch = []

            self.processBatch(currentBatch)
            inputfile.close()

    def processBatch(self, currentBatch):

