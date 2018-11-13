# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 19:10:09 2018

@author: LENOVO
"""

import numpy as np
import unittest
from Convolution import convolution

class testFunction(unittest.TestCase):
    def testConvolution(self):
        image = np.array([[1.0,0.0,1.0,0.0,1.0], [1.0,0.0,1.0,0.0,1.0], [1.0,0.0,1.0,0.0,1.0], [1.0,0.0,1.0,0.0,1.0], [1.0,0.0,1.0,0.0,1.0]])
        convolutionFilter = np.array([[1,4,7,4,1],[4,16,26,16,4],[7,26,41,26,7],[4,16,26,16,4],[1,4,7,4,1]])/271.0
        filteredImage = convolution(convolutionFilter, image)
        
        self.assertAlmostEquals(filteredImage[0,0], 141.0/271.0)
    
if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(testFunction)
    unittest.TextTestRunner(verbosity=2).run(suite)