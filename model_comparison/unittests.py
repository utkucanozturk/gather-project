# if needed install packages
#!pip install -r requirements.txt --use-deprecated=legacy-resolver

import unittest
import model_comparison
import numpy as np
import os
HOME = r"/content/drive/MyDrive/gather/ML-DL-scripts/DEEP LEARNING/segmentation/Segmentation pipeline"
os.chdir(HOME)


class Test_TestIoUMetric(unittest.TestCase):

    def test_IoIMetric(self):

        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0])
        y_true = y_true.reshape((3, 3))
        y_predicted = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1])
        y_predicted = y_predicted.reshape((3, 3))

        TP = 2
        FP = 3
        FN = 2
        jaccard = TP / (TP+FP + FN)
        jaccard

        calculated_iou = model_comparison.iou_metric(
            y_true, y_predicted)[0][0][0]
        true_iou = jaccard
        EPSILON = 0.001
        self.assertTrue(abs(calculated_iou - true_iou) <= EPSILON)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
