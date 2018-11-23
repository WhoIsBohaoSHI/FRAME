# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 19:10:09 2018

@author: LENOVO
"""

import cv2
import random
import numpy as np
import FRAME
import scipy.stats as ss

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
            pval = np.array([self.densityFunction(self.convolutionFilters, lambdaParameter, modifyImage(synthesizedImage, randomIndex, j)) for j in range(self.numOfGreyLevel)])
            pval /= pval.sum()
            print(pval)
            greyLevel = np.argmax(np.random.multinomial(1, pval))
            synthesizedImage = modifyImage(synthesizedImage, randomIndex, greyLevel) 
            print(i)
            
        return synthesizedImage
    
    
class ImportanceSamplerForFrame():
    def __init__(self, convolutionFilters, numOfSamples, numOfGreyLevel, histgramLevel):
        self.convolutionFilters = convolutionFilters
        self.numOfSamples = numOfSamples
        self.numOfGreyLevel = numOfGreyLevel
        self.histgramLevel = histgramLevel
        
    def densityFunction(self, filters, Lambda, image):
        numOfFilters = len(filters)
        innerProduct = np.array([np.inner(Lambda[i], cv2.calcHist(cv2.filter2D(image,-1,filters[i]),channels=[0], mask = None, histSize = [self.histgramLevel], ranges=[0,256]).reshape([self.histgramLevel,])) for i in range(numOfFilters)])
        return np.exp(-(innerProduct.sum()))
        
    def __call__(self, imageX, imageY, lambdaParameter):
        imageSamples = []
        weights = []
        for i in range(self.numOfSamples):
            sample = creatWhiteNoiseImage(imageX, imageY)
            imageSamples.append(sample)
            weights.append(self.densityFunction(self.convolutionFilters, lambdaParameter, sample))
            print (i)
        imageSamples = np.array(imageSamples).reshape([self.numOfSamples, -1])
        weights = normalize(np.array(weights))
        synthesizedImage = np.array([imageSamples[i] * weights[i] for i in range(self.numOfSamples)])
        print((np.sum(synthesizedImage,0).astype(int)).reshape([imageX, imageY]))
        return (np.sum(synthesizedImage,0).astype(int)).reshape([imageX, imageY])