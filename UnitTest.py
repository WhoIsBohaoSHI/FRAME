# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 19:10:09 2018

@author: LENOVO
"""

import numpy as np
import unittest
from Convolution import *

class testFunction(unittest.TestCase):
    def testModifyImage(self):
        image = np.array([[1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0], [1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0],[1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0],[1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0],
                          [1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0],[1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0],[1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0],[1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0]])
        modifiedImage = modifyImage(image, [0,0], 0.5)    
        self.assertAlmostEquals(modifiedImage[0,0], 0.5)
        
    def testConvolution(self):
        image = np.array([[1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0], [1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0],[1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0],[1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0],
                          [1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0],[1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0],[1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0],[1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0]])
        convolutionFilters = [np.array([[1,4,7,4,1],[4,16,26,16,4],[7,26,41,26,7],[4,16,26,16,4],[1,4,7,4,1]])/271.0]
        convolutionedImage = convolutionFunction(convolutionFilters, image)[0]
        self.assertAlmostEquals(convolutionedImage[2,2], 141.0/271.0)
        self.assertAlmostEquals(convolutionedImage[3,3], 132.0/271.0)
        
    def testComputeHistogram(self):
        image = np.array([[1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0], [1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0],[1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0],[1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0],
                          [1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0],[1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0],[1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0],[1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0]])
        histogram = computeHistogram(image, 32)
        self.assertAlmostEquals(histogram[0], 0.5)
        for i in range(30):
            self.assertAlmostEquals(histogram[i+1], 0.0)
        self.assertAlmostEquals(histogram[31], 0.5)
        self.assertAlmostEquals(histogram.sum(), 1.0)
        self.assertEquals(len(histogram), 32)
        
        histogram = computeHistogram(image, 32, [0.1, 0.9])
        for i in range(32):
            self.assertAlmostEquals(histogram[i], 0.0)
    
    def testCreatWhiteNoiseImage(self):
        image = np.array([[1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0], [1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0],[1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0],[1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0],
                          [1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0],[1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0],[1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0],[1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0]])
        
        whiteNoiseImage = creatWhiteNoiseImage(8,8)    
        self.assertTrue(np.all(whiteNoiseImage >= 0))
        self.assertTrue(np.all(whiteNoiseImage <= 255))
        self.assertEqual(whiteNoiseImage.shape[0], 8)
        self.assertEqual(whiteNoiseImage.shape[1], 8)
        
    def testNormalize(self):
        vector1 = np.array([0.1,0.25,0.15])
        vector2 = np.array([10,25,15])
        self.assertAlmostEquals(normalize(vector1).sum(), 1.0)
        self.assertAlmostEquals(normalize(vector1)[0], 0.2)
        self.assertAlmostEquals(normalize(vector2).sum(), 1.0)
        self.assertAlmostEquals(normalize(vector2)[1], 0.5)
        
    def testEuclideanDistance(self):
        vector1 = np.array([0, 0, 0])
        vector2 = np.array([0, 1, -1])
        vector3 = np.array([0, -1, 1])
        self.assertAlmostEquals(euclideanDistance(vector1, vector2), np.sqrt(2))
        self.assertAlmostEquals(euclideanDistance(vector1, vector3), np.sqrt(2))
        self.assertAlmostEquals(euclideanDistance(vector2, vector3), np.sqrt(8))
        
    def innerProduct(self):
        vector1 = np.array([0, 0, 0])
        vector2 = np.array([0, 1, -1])
        vector3 = np.array([0, -1, 1])
        self.assertAlmostEquals(np.inner(vector1, vector2), 0)
        self.assertAlmostEquals(np.inner(vector1, vector3), 0)
        self.assertAlmostEquals(np.inner(vector2, vector3), -2)
        
if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(testFunction)
    unittest.TextTestRunner(verbosity=2).run(suite)