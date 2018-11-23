# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 19:10:09 2018

@author: LENOVO
"""

import cv2
import numpy as np
import os
import scipy.stats as ss


def creatWhiteNoiseImage(imageX, imageY):
    randomByteArray = bytearray(os.urandom(imageX * imageY))
    whiteNoise = np.array(randomByteArray).reshape([imageX, imageY]) 
    return whiteNoise       
        
def modifyImage(image, modyfyIndex, pixalValue):
    image[modyfyIndex[0],modyfyIndex[1]] = pixalValue
    return image

def euclideanDistance(vector1, vector2):
    distance = np.sqrt(((vector1 - vector2) ** 2).sum())
    return distance

def normalize(vector):
    return vector/(np.sum(vector) + 1e-10)

def convolutionFunction(convolutionFilters, image):
    convolutionedImage = np.array([cv2.filter2D(image, -1, convolutionFilters[i]) for i in range(len(convolutionFilters))])
    return convolutionedImage

def computeHistogram(image, histogramLevel = 32, histogramRange = None):
    totalPixal = image.shape[0] * image.shape[1]
    if histogramRange == None:
        histogramRange = [np.min(image), np.max(image) + 1e-10]
    bandWidth = (histogramRange[1] - histogramRange[0]) / histogramLevel
    histRange = [histogramRange[0] + i * bandWidth for i in range(histogramLevel+1)]
    frequency = [totalPixal - ((image < histRange[i]).sum() + (image >= histRange[i+1]).sum()) for i in range(histogramLevel)]
    histogram = normalize(np.array(frequency))
    return histogram
    
    