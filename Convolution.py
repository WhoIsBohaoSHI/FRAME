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
    return np.sqrt(((vector1 - vector2) ** 2).sum())

def normalize(vector):
    return vector/np.sum(vector)

def convolutionAndComputeHistgram(histgramLevel, convolutionFilters, image):
    imageX = image.shape[0]
    imageY = image.shape[1]
    histgrams = np.array([cv2.calcHist(cv2.filter2D(image.reshape(1, imageX, imageY), -1, convolutionFilters[i]), channels=[0], mask = None, histSize = [histgramLevel], ranges=[-1.0,256.5]) for i in range(len(convolutionFilters))]).reshape([histgramLevel, -1])
    normalizedHistgrams = np.array([histgrams[:,i]/histgrams[:,i].sum() for i in range(len(convolutionFilters))]).T
    return normalizedHistgrams