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
    
    def __call__(self, histgramLevel, sampler, epsilon):
        numOfFilters = len(self.convolutionFilters)
        imageX = self.observedImage.shape[0]
        imageY = self.observedImage.shape[1]
        observedHistgrams = cvl.convolutionAndComputeHistgram(histgramLevel, self.convolutionFilters, self.observedImage)
        lambdaParameter = np.zeros([numOfFilters, histgramLevel])
        synthesizedImage = cvl.creatWhiteNoiseImage(imageX, imageY)
        synthesizedHistgrams = cvl.convolutionAndComputeHistgram(histgramLevel, self.convolutionFilters, synthesizedImage)

        while all(np.array([cvl.euclideanDistance(synthesizedHistgrams[:,i], observedHistgrams[:,i]) for i in range(numOfFilters)]) > epsilon):
            deltaLambda = np.array([synthesizedHistgrams[:,i] - observedHistgrams[:,i] for i in range(numOfFilters)])
            lambdaParameter = np.array([lambdaParameter[i] + deltaLambda[i] for i in range(numOfFilters)])       
            synthesizedImage = sampler(synthesizedImage, lambdaParameter)
            synthesizedHistgrams = self.computeHistgram(histgramLevel, self.convolutionFilters, synthesizedImage)
            
        return lambdaParameter, synthesizedImage
    
class GibbsSamplerForFrame():
    def __init__(self, convolutionFilters, numOfGibbsSweeps, numOfGreyLevel, histgramLevel):
        self.convolutionFilters = convolutionFilters
        self.numOfGibbsSweeps = numOfGibbsSweeps
        self.numOfGreyLevel = numOfGreyLevel
        self.histgramLevel = histgramLevel
        
    def densityFunction(self, filters, Lambda, image):
        numOfFilters = len(filters)
        innerProduct = np.array([np.inner(Lambda[i], cv2.calcHist(cv2.filter2D(image,-1,filters[i]),channels=[0], mask = None, histSize = [self.histgramLevel], ranges=[0,256]).reshape([self.histgramLevel,])) for i in range(numOfFilters)])
        return np.exp(-(innerProduct.sum()))
        
    def __call__(self, synthesizedImage, lambdaParameter):
        for i in range(synthesizedImage.shape[0] * synthesizedImage.shape[1] * self.numOfGibbsSweeps):
            randomIndex = [random.randint(0, synthesizedImage.shape[0] - 1), random.randint(0, synthesizedImage.shape[1] - 1)]
            pval = np.array([self.densityFunction(self.convolutionFilters, lambdaParameter, cvl.modifyImage(synthesizedImage, randomIndex, j)) for j in range(self.numOfGreyLevel)])
            pval /= pval.sum()
            print(pval)
            greyLevel = np.argmax(np.random.multinomial(1, pval))
            synthesizedImage = cvl.modifyImage(synthesizedImage, randomIndex, greyLevel) 
            print(i)
            
        return synthesizedImage


if __name__ == "__main__" :  
    numOfObservedImages = 3
    numOfFilters = 5
    histSize = 64
    observedImages = np.array([cv2.imread('tex%d.jpg'%(i+1), 0) for i in range(numOfObservedImages)])
        
    averageFilter = np.ones([5,5])/25.0
    gaussianFilter = np.array([[1,4,7,4,1],[4,16,26,16,4],[7,26,41,26,7],[4,16,26,16,4],[1,4,7,4,1]])/271.0
    laplaceFilter = np.array([[0,0,1,0,0],[0,1,2,1,0],[1,2,-16,2,1],[0,1,2,1,0],[0,0,1,0,0]])/16.0
    gaborFilter_1 = cv2.getGaborKernel((5, 5), 1.0, np.pi/4, np.pi/2 , 0.5, 0) 
    gaborFilter_2 = cv2.getGaborKernel((5, 5), 1.0, 3*np.pi/4, np.pi/2 , 0.5, 0)
    gaborFilter_1 /= gaborFilter_1.sum()
    gaborFilter_2 /= gaborFilter_2.sum()
    filters = [averageFilter, gaussianFilter, laplaceFilter, gaborFilter_1, gaborFilter_2]
    
    frameModel = FrameModel(filters, observedImages[0])
    sampler = GibbsSamplerForFrame(filters, 1, 256, 32)
    #sampler = ImportanceSamplerForFrame(filters, 10000, 256, 32)
    lambdaParameter, synthesizedImage = frameModel(32, sampler, 0.05)
    
