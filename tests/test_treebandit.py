# -*- coding: utf-8 -*-

import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from mabwiser.mab import LearningPolicy
from tests.test_base import BaseTest


class TreeBanditTest(BaseTest):
    # TODO: tests will be changed after adding in mab_policy
    def test_treebandit(self):
        context_history = np.array([[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                    [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                    [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                    [0, 2, 1, 0, 0]])

        arm, mab = self.predict(arms=[1, 2, 3],
                                decisions=[1, 1, 1, 2, 2, 2, 3, 3, 3, 1],
                                rewards=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                                learning_policy=LearningPolicy.TreeBandit(),
                                context_history=context_history,
                                contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                seed=123456,
                                num_run=3,
                                is_predict=True)

        self.assertEqual(len(arm), 3)
        self.assertEqual(arm, [[1, 1], [1, 1], [1, 1]])

        """
        arm, mab = self.predict(arms=['Arm1', 'Arm2'],
                                decisions=['Arm1', 'Arm1', 'Arm2', 'Arm1'],
                                rewards=[20, 17, 25, 9],
                                learning_policy=LearningPolicy.TreeBandit(),
                                context_history=[[0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 1, 0], [3, 2, 1, 0]],
                                contexts=[[2, 3, 1, 0]],
                                seed=123456,
                                num_run=1,
                                is_predict=True)
        """

    def test_treebandit_expectations(self):
        exps, mab = self.predict(arms=[1, 2, 3],
                                 decisions=[1, 1, 1, 2, 2, 2, 3, 3, 3, 1],
                                 rewards=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                                 learning_policy=LearningPolicy.TreeBandit(),
                                 context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                  [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                  [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                  [0, 2, 1, 0, 0]],
                                 contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                 seed=123456,
                                 num_run=1,
                                 is_predict=False)

        self.assertListAlmostEqual(exps[0].values(), [1, 1, 1])
        self.assertListAlmostEqual(exps[1].values(), [1, 1, 1])
