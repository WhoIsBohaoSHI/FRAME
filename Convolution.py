# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 19:10:09 2018

@author: LENOVO
"""

import cv2
import numpy as np


def convolution(convolutionFilter, image):
    filteredImage = cv2.filter2D(image, -1, convolutionFilter)
    return filteredImage