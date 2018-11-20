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
        
    def testConvolutionAndComputeHistgram(self):
        image = np.array([[1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0], [1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0],[1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0],[1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0],
                          [1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0],[1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0],[1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0],[1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0]]).astype(int)*255
        convolutionFilter = np.array([1.0,4.0,7.0,4.0,1.0], )
        histogram = convolutionAndComputeHistgram(32, image, convolutionFilter)
        print(histogram)
        
        self.assertAlmostEqual(histogram.sum(), 1.0)
        self.assertAlmostEqual(histogram[0], 0.5)
        
if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(testFunction)
    unittest.TextTestRunner(verbosity=2).run(suite)