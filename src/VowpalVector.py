#!/usr/bin/env python

import mmhash

class VowpalVector:

    def __init__(self, nbits=18, lamb=.1):
        self.scaling = 1.
        self.nbits = nbits
        self.lamb = lamb
        self.biasKey = "BIAS"

        self.vectorSize = 2 ** nbits
        self.resetWeights()

    def resetWeights(self):
        self.weights = [0. for x in range(0, self.vectorSize)]

    def innerProduct(self, vec):
        """ Input format here is similar to Vowpal Wabbit:
            <feature_name>:<feature_value> """

        # Bias term always takes a constant value of 1 for all our observations
        # e.g., intercept in linear regression.
        innerProd = (self.weights[self._hash(self.biasKey)] *
                     self._rademacher(self.biasKey))

        for coord in vec:
            feature = coord.split(':')
            featureName = str(feature[0])
            featureVal = float(feature[1])
            
            wtVal = self.weights[self._hash(featureName)]
            radVal = self._rademacher(featureName)
            innerProd = innerProd + (featureVal * wtVal * radVal)

        return innerProd

    def processBatch(self, currentBatch, t):
        k = len(currentBatch)
        updateSet = []
        # Determine which items in batch contribute to loss
        for obs in currentBatch:
            if len(obs) < 2: continue
            y = int(obs[0])

            if y != -1 and y != 1:
                print("Disregarding observation because of invalid training \
                      observation.  Labels must be either -1 or 1")
                continue

            features = obs[1:]
            inner_product = self.innerProduct(features)

            if y * inner_product < 1:
                updateSet.append(obs)
        
        # Update weight vector using the update set
        self.updateWeights(self.lamb, t, k, updateSet)

    def updateWeights(self, lamb, t, k, updateSet):
        stepKeys = self.gradientStep(lamb, t, k, updateSet)
        self.projectStep(lamb, stepKeys)

    def gradientStep(self, lamb, t, k, updateSet):
        eta = 1./(lamb * t)
        stepDirection = {}

        # Initialize stepDirection with key corresponding to bias term

        stepDirection[self.biasKey] = 0
        # Compute the subgradient of our loss function at the present update set
        for obs in updateSet:
            if len(obs) < 2: continue

            y = int(obs[0])

            # Compute subgradient of bias(intercept) term
            stepDirection[self.biasKey] += y

            # Compute subgradient of features
            features = obs[1:]
            for coord in features:
                feature = coord.split(':')
                featureName = str(feature[0])
                featureVal = float(feature[1]) * y
                if featureName not in stepDirection:
                    stepDirection[featureName] = 0.

                stepDirection[featureName] += featureVal

        scaling = eta/k

        # Update weight coefficients
        for key,val in stepDirection.iteritems():
            indx = self._hash(key)
            rad = self._rademacher(key)

            self.weights[indx] *= (1 - (eta * lamb))
            self.weights[indx] += scaling * val * rad
        
        return stepDirection.keys()

    def projectStep(self, lamb, stepKeys):

        normSquared = 0.

        for feature in stepKeys:
            indx = self._hash(feature)
            normSquared += self.weights[indx] ** 2

        scaling = 1
        if normSquared != 0:
            scaling = min(1, (1/(lamb ** .5))/(normSquared ** .5))

        if scaling != 1:
            for feature in stepKeys:
                indx = self._hash(feature)
                self.weights[indx] *= scaling
            

    def _hash(self, val):
        return mmhash.get_hash(val) % self.vectorSize

    def _rademacher(self, val):
        if (mmhash.get_hash(val) >> 1) % 2 == 0:
            return 1 
        else:
            return -1

    def _norm(self):
        sqNorm = sum(map(lambda x: x*x, self.weights))
        return sqNorm ** .5
