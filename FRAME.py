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
    
    def __call__(self, histgramLevel, sampler, epsilon = 0.001, isSave = 1, synthesizedImageInit = None, lambdaParameterInit = None):
        numOfFilters = len(self.convolutionFilters)
        imageX = self.observedImage.shape[0]
        imageY = self.observedImage.shape[1]
        convolutionedObservedImage = np.array(cvl.convolutionFunction(self.convolutionFilters, self.observedImage))
        observedHistgrams = np.array([cvl.computeHistogram(convolutionedObservedImage[i], histgramLevel) for i in range(numOfFilters)])
#        histgramRange = [np.min(self.observedImage), np.max(self.observedImage) + 1e-10]
        if isSave == 0:
            lambdaParameter = [np.zeros([numOfFilters, histgramLevel])]   
            synthesizedImage = cvl.creatWhiteNoiseImage(imageX, imageY)
        else:
            lambdaParameter = [lambdaParameterInit]
            synthesizedImage = synthesizedImageInit
        convolutionedSynthesizedImage = np.array(cvl.convolutionFunction(self.convolutionFilters, synthesizedImage))
        synthesizedHistgrams = np.array([cvl.computeHistogram(convolutionedSynthesizedImage[i], histgramLevel) for i in range(numOfFilters)])
        print(observedHistgrams,synthesizedHistgrams)
        j = 0
        while (j < 3000) and (any(np.array([cvl.euclideanDistance(synthesizedHistgrams[i], observedHistgrams[i]) for i in range(numOfFilters)]) > epsilon)):
            deltaLambda = synthesizedHistgrams - observedHistgrams
            lambdaParameter.append(np.array(lambdaParameter[-1] + deltaLambda))     
            print(deltaLambda)
            synthesizedImage = sampler(synthesizedImage, lambdaParameter)
            convolutionedSynthesizedImage = np.array(cvl.convolutionFunction(self.convolutionFilters, synthesizedImage))
            synthesizedHistgrams = np.array([cvl.computeHistogram(convolutionedSynthesizedImage[i], histgramLevel) for i in range(numOfFilters)])
            print(j)
            j = j + 1
         
        return lambdaParameter, synthesizedImage
    
class GibbsSamplerForFrame():
    def __init__(self, convolutionFilters, filterSize, histgramLevel, numOfGibbsSweeps = 4, numOfGreyLevel = 8):
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
            pval = np.array([self.densityFunction(self.convolutionFilters, lambdaParameter, cvl.modifyImage(synthesizedImage, randomIndex, (j + 0.5) * (256./self.numOfGreyLevel))[modifiedRange[0]:modifiedRange[1],modifiedRange[2]:modifiedRange[3]], self.histgramLevel) for j in range(self.numOfGreyLevel)])
#            print(pval)
            pval /= pval.sum()
            
            greyLevel = np.argmax(np.random.multinomial(1, pval))
            synthesizedImage = cvl.modifyImage(synthesizedImage, randomIndex, np.round(np.random.uniform(greyLevel * (256./self.numOfGreyLevel), (greyLevel+1) * (256./self.numOfGreyLevel)))) 
#            print(np.round(np.random.uniform(greyLevel * (256./self.numOfGreyLevel), (greyLevel+1) * (256./self.numOfGreyLevel))))
        print(pval)
    #        print(i)
            
        return synthesizedImage
    
class FeaturePursuit():
    def __init__(self, filtersBank, observedImage, numOfSelectedFeature):
        self.filtersBank = filtersBank
        self.observedImage = observedImage
        self.numOfSelectedFeature = numOfSelectedFeature
    
    def __call__(self, histgramLevel, epsilon = 0.001):
        numOfFilters = len(self.filtersBank)
        filtersIndex = [i for i in range(numOfFilters)]
        selectedFeature = []
        selectedIndex = []
        k = 0
        imageX = self.observedImage.shape[0]
        imageY = self.observedImage.shape[1]
        convolutionedObservedImage = np.array(cvl.convolutionFunction(self.filtersBank, self.observedImage))
        observedHistgrams = np.array([cvl.computeHistogram(convolutionedObservedImage[i], histgramLevel) for i in range(numOfFilters)])
        synthesizedImage = cvl.creatWhiteNoiseImage(imageX, imageY)
        lambdaParameter  =  np.array([])
        
        distance = 1.
        while (distance > epsilon) and (k < self.numOfSelectedFeature):
            indexSet = [index for index in filtersIndex if index not in selectedIndex]
            featureSet = [self.filtersBank[index] for index in indexSet]
            numOfFeatures = len(featureSet)
            convolutionedSynthesizedImage = np.array(cvl.convolutionFunction(featureSet, synthesizedImage))
            synthesizedHistgrams = np.array([cvl.computeHistogram(convolutionedSynthesizedImage[i], histgramLevel) for i in range(numOfFeatures)])
            featureDistances = np.array([cvl.euclideanDistance(observedHistgrams[indexSet[i]], synthesizedHistgrams[i]) for i in range(numOfFeatures)])
            print(featureDistances)
            maxIndex = np.argmax(featureDistances, 0)
            selectedFeature.append(featureSet[maxIndex])
            selectedIndex.append(indexSet[maxIndex])
            distance = np.max(featureDistances)
            lambdaList = list(lambdaParameter)
            lambdaList.append(np.zeros(histgramLevel))
            lambdaParameter = np.array(lambdaList)

            k = k + 1
            print('get feature', k, indexSet[maxIndex])
            frameModel = FrameModel(selectedFeature, self.observedImage)
            featureSize = selectedFeature[0].shape[0]
            sampler = GibbsSamplerForFrame(selectedFeature, featureSize, histgramLevel)
            lambdas, synthesizedImage = frameModel(histgramLevel, sampler, isSave = 1, synthesizedImageInit = synthesizedImage, lambdaParameterInit = lambdaParameter)
            lambdaParameter = lambdas[-1]
        return selectedFeature, selectedIndex, synthesizedImage

if __name__ == "__main__" :  
    numOfObservedImages = 1
        
    averageFilter = np.ones([5,5])/25.0
    gaussianFilter = np.array([[1,4,7,4,1],[4,16,26,16,4],[7,26,41,26,7],[4,16,26,16,4],[1,4,7,4,1]])/271.0
    laplaceFilter = np.array([[0,0,1,0,0],[0,1,2,1,0],[1,2,-16,2,1],[0,1,2,1,0],[0,0,1,0,0]])/16.0
    gaborFilter_1 = cv2.getGaborKernel((5, 5), 1.0, np.pi/4, np.pi/2 , 0.5, 0) 
    gaborFilter_2 = cv2.getGaborKernel((5, 5), 1.0, 3*np.pi/4, np.pi/2 , 0.5, 0)
    gaborFilter_1 /= gaborFilter_1.sum()
    gaborFilter_2 /= gaborFilter_2.sum()
#    filters = [averageFilter, gaussianFilter, laplaceFilter, gaborFilter_1, gaborFilter_2]
    filters = [averageFilter]
    
    observedImages = np.array([cv2.imread('tex%d.jpg'%(i), 0) for i in range(numOfObservedImages)])
#    observedImages = np.zeros([1,40,40])
#    for i in range(8):
#        for j in range(8):
#            observedImages[0,5*i:5*(i+1),5*j:5*(j+1)] = gaussianFilter * 271.0
#    observedImages = observedImages * 255.0 / 41.0
    cv2.imshow('synthesized',observedImages[0]/255.0)
    cv2.waitKey(0)
    print('ok')
    
    #FRAME algorithm 1 :
    
    frameModel = FrameModel(filters, observedImages[0])
    sampler = GibbsSamplerForFrame(filters, 5, 8, 4, 8)
    lambdaParameter, synthesizedImage = frameModel(8, sampler, 0.001, 0)
    cv2.imshow('synthesized',synthesizedImage/255.0)
    cv2.waitKey(0)
    
    #FRAME algorithm 3 : feature pursuit
    
#    featurePursuit = FeaturePursuit(filters, observedImages[0], 3)
#    selectedFeature, selectedIndex, synthesizedImage = featurePursuit(8, 0.001)
#    cv2.imshow('synthesized',synthesizedImage/255.0)
#    cv2.waitKey(0)
#    
#    
    
    