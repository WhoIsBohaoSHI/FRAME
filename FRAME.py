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

class FrameModel():
    def __init__(self, convolutionFilters, observedImage):
        self.convolutionFilters = convolutionFilters
        self.observedImage = observedImage
    
    def __call__(self, histgramLevel, sampler, epsilon, isSave = 1, synthesizedImage = None, lambdaParameter = None):
        numOfFilters = len(self.convolutionFilters)
        imageX = self.observedImage.shape[0]
        imageY = self.observedImage.shape[1]
        histgramRange = [0., 255.]
        convolutionedObservedImage = np.array(cvl.convolutionFunction(self.convolutionFilters, self.observedImage))
        observedHistgrams = np.array([cvl.computeHistogram(convolutionedObservedImage[i], histgramLevel, histgramRange) for i in range(numOfFilters)])
#        histgramRange = [np.min(self.observedImage), np.max(self.observedImage) + 1e-10]
        if isSave == 0:
            lambdaParameter = [np.zeros([numOfFilters, histgramLevel])]   
            synthesizedImage = cvl.creatWhiteNoiseImage(imageX, imageY)
        convolutionedSynthesizedImage = np.array(cvl.convolutionFunction(self.convolutionFilters, synthesizedImage))
        synthesizedHistgrams = np.array([cvl.computeHistogram(convolutionedSynthesizedImage[i], histgramLevel) for i in range(numOfFilters)])
        print(observedHistgrams,synthesizedHistgrams)
        j = 0
        while (j < 2000) and (any(np.array([cvl.euclideanDistance(synthesizedHistgrams[i], observedHistgrams[i]) for i in range(numOfFilters)]) > epsilon)):
            deltaLambda = synthesizedHistgrams - observedHistgrams
            lambdaParameter.append(np.array(lambdaParameter[-1] + deltaLambda))     
            print(lambdaParameter[-1][0])
            synthesizedImage = sampler(synthesizedImage, lambdaParameter)
            convolutionedSynthesizedImage = np.array(cvl.convolutionFunction(self.convolutionFilters, synthesizedImage))
            synthesizedHistgrams = np.array([cvl.computeHistogram(convolutionedSynthesizedImage[i], histgramLevel) for i in range(numOfFilters)])
            print(j)
            j = j + 1
         
        return lambdaParameter, synthesizedImage
    
class GibbsSamplerForFrame():
    def __init__(self, convolutionFilters, filterSize, numOfGibbsSweeps, numOfGreyLevel, histgramLevel):
        self.convolutionFilters = convolutionFilters
        self.filterSize = filterSize
        self.numOfGibbsSweeps = numOfGibbsSweeps
        self.numOfGreyLevel = numOfGreyLevel
        self.histgramLevel = histgramLevel
        
    def densityFunction(self, filters, Lambda, image, histgramLevel):
        numOfFilters = len(filters)
        convolutionedImage = np.array(cvl.convolutionFunction(filters, image))
        histograms = np.array([cvl.computeHistogram(convolutionedImage[i], histgramLevel) for i in range(numOfFilters)])
        innerProduct = np.array([np.inner(Lambda[-1][i], histograms[i]) for i in range(numOfFilters)])
        return np.exp(-(innerProduct.sum()))
        
    def __call__(self, synthesizedImage, lambdaParameter):
        for i in range(synthesizedImage.shape[0] * synthesizedImage.shape[1] * self.numOfGibbsSweeps):
            randomIndex = [random.randint(0, synthesizedImage.shape[0] - 1), random.randint(0, synthesizedImage.shape[1] - 1)]
            modifiedRange = [max(0,randomIndex[0]-self.filterSize),min(synthesizedImage.shape[0],randomIndex[0]+self.filterSize), max(0,randomIndex[1]-self.filterSize),min(synthesizedImage.shape[1], randomIndex[1]+self.filterSize)]
            pval = np.array([self.densityFunction(self.convolutionFilters, lambdaParameter, cvl.modifyImage(synthesizedImage, randomIndex, (j + 0.5) * (256./self.numOfGreyLevel))[modifiedRange[0]:modifiedRange[1],modifiedRange[2]:modifiedRange[3]], 32) for j in range(self.numOfGreyLevel)])
            pval /= pval.sum()
            
            greyLevel = np.argmax(np.random.multinomial(1, pval))
            synthesizedImage = cvl.modifyImage(synthesizedImage, randomIndex, np.round(np.random.uniform(greyLevel * (256./self.numOfGreyLevel), (greyLevel+1) * (256./self.numOfGreyLevel)))) 
#            print(np.round(np.random.uniform(greyLevel * (256./self.numOfGreyLevel), (greyLevel+1) * (256./self.numOfGreyLevel))))
        print(pval)
    #        print(i)
            
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
    filters = [gaussianFilter]
    
#    observedImages = np.array([cv2.imread('tex%d.jpg'%(i), 0) for i in range(numOfObservedImages)])
    observedImages = np.zeros([1,40,40])
    for i in range(8):
        for j in range(8):
            observedImages[0,5*i:5*(i+1),5*j:5*(j+1)] = gaussianFilter * 271.0
    observedImages = observedImages * 255.0 / 41.0
    
    cv2.imshow('synthesized',observedImages[0]/255.0)
    cv2.waitKey(0)
    print('ok')
    frameModel = FrameModel(filters, observedImages[0])
    sampler = GibbsSamplerForFrame(filters, 5, 4, 8, 32)
    #sampler = ImportanceSamplerForFrame(filters, 10000, 256, 32)
    lambdaParameter, synthesizedImage = frameModel(32, sampler, 0.001, 1, synthesizedImage, lambdaParameter)
    cv2.imshow('synthesized',synthesizedImage/255.0)
    cv2.waitKey(0)
    
    
    
    