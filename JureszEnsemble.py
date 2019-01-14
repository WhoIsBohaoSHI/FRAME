# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 19:10:09 2018

@author: LENOVO
"""

import Convolution as cvl
import matplotlib.pyplot as plt
import cv2
import operator as op
import random
import numpy as np
import os
import scipy.stats as ss


    
class GibbsSamplerForJureszEnsemble():
    def __init__(self, observedImages, convolutionFilters, filterSize, numOfGreyLevel, histgramLevel, numOfGibbsSweeps):
        self.convolutionFilters = convolutionFilters
        self.numOfFilters = len(convolutionFilters)
        self.observedImages = observedImages
        self.filterSize = filterSize
        self.numOfGreyLevel = numOfGreyLevel
        self.histgramLevel = histgramLevel
        self.numOfGibbsSweeps = numOfGibbsSweeps
        
    def gFunction(self, observedHistogram, synthesizedHistogram, distanceFunction, epsilon):
        distance = np.array([distanceFunction(observedHistogram[i], synthesizedHistogram[i]) for i in range(self.numOfFilters)])
        if all(distance <= epsilon):
            G = 0
        else:
            G = distance.sum()
        return G
        
    def densityFunction(self, filters, synthesizedImage, histgramLevel, temperature):
        convolutionedObservedImage = np.array(cvl.convolutionFunction(filters, self.observedImages))
        observedHistograms = np.array([cvl.computeHistogram(convolutionedObservedImage[i], histgramLevel) for i in range(self.numOfFilters)])
        convolutionedSynthesizedImage = np.array(cvl.convolutionFunction(filters, synthesizedImage))
        synthesizedHistograms = np.array([cvl.computeHistogram(convolutionedSynthesizedImage[i], histgramLevel) for i in range(self.numOfFilters)])
        G = self.gFunction(observedHistograms, synthesizedHistograms, cvl.euclideanDistance, 0.001)
        
        return np.exp(-G/temperature)
        
    def __call__(self, initialTemperature, thesold):
        imageX = self.observedImages.shape[0]
        imageY = self.observedImages.shape[1]
        synthesizedImage = cvl.creatWhiteNoiseImage(imageX, imageY)
        temperature = initialTemperature
        t = 0

        while temperature > thesold:
            print(t, temperature)
            for i in range(imageX * imageY * self.numOfGibbsSweeps):
                randomIndex = [random.randint(0, synthesizedImage.shape[0] - 1), random.randint(0, synthesizedImage.shape[1] - 1)]
                modifiedRange = [max(0,randomIndex[0]-self.filterSize),min(synthesizedImage.shape[0],randomIndex[0]+self.filterSize), max(0,randomIndex[1]-self.filterSize),min(synthesizedImage.shape[1], randomIndex[1]+self.filterSize)]
                pval = np.array([self.densityFunction(self.convolutionFilters, cvl.modifyImage(synthesizedImage, randomIndex, (j + 0.5) * (256./self.numOfGreyLevel))[modifiedRange[0]:modifiedRange[1],modifiedRange[2]:modifiedRange[3]],
                                self.histgramLevel, temperature) for j in range(self.numOfGreyLevel)])
                pval /= pval.sum()
                greyLevel = np.argmax(np.random.multinomial(1, pval))
                synthesizedImage = cvl.modifyImage(synthesizedImage, randomIndex, np.round(np.random.uniform(greyLevel * (256./self.numOfGreyLevel), (greyLevel+1) * (256./self.numOfGreyLevel)))) 
            t = t + 1
            temperature = initialTemperature / np.log(1 + t)
            print(pval)
            
        return synthesizedImage


if __name__ == "__main__" :  
    numOfObservedImages = 1
#    observedImages = np.zeros([1,128,128])
        
    averageFilter = np.ones([5,5])/25.0
    gaussianFilter = np.array([[1,4,7,4,1],[4,16,26,16,4],[7,26,41,26,7],[4,16,26,16,4],[1,4,7,4,1]])/271.0
    laplaceFilter = np.array([[0,0,1,0,0],[0,1,2,1,0],[1,2,-16,2,1],[0,1,2,1,0],[0,0,1,0,0]])/16.0
    gaborFilter_1 = cv2.getGaborKernel((5, 5), 1.0, np.pi/4, np.pi/2 , 0.5, 0) 
    gaborFilter_2 = cv2.getGaborKernel((5, 5), 1.0, 3*np.pi/4, np.pi/2 , 0.5, 0)
    gaborFilter_1 /= gaborFilter_1.sum()
    gaborFilter_2 /= gaborFilter_2.sum()
#    filters = [averageFilter, gaussianFilter, laplaceFilter, gaborFilter_1, gaborFilter_2]
    filters = [averageFilter]
    
#    observedImages = np.array([cv2.imread('tex%d.jpg'%(i), 0) for i in range(numOfObservedImages)])
    observedImages = np.zeros([1,40,40])
    for i in range(8):
        for j in range(8):
            observedImages[0,5*i:5*(i+1),5*j:5*(j+1)] = gaussianFilter * 271.0
    observedImages = observedImages * 255.0 / 41.0
    
    cv2.imshow('synthesized',observedImages[0]/255.0)
    cv2.waitKey(0)
    print('ok')
    jureszEnsemble = GibbsSamplerForJureszEnsemble(observedImages[0], filters, 5, 8, 32, 2)
    synthesizedImage = jureszEnsemble(10.0, 1.25)
    cv2.imshow('synthesized',synthesizedImage/255.0)
    cv2.waitKey(0)
    
    
    
    