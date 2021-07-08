# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, NoReturn, Optional, Callable

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from mabwiser.base_mab import BaseMAB
from mabwiser.utils import reset, argmax, Arm, Num, _BaseRNG


class _TreeBandit(BaseMAB):
    # TODO: set default for mab_policy
    def __init__(self, rng: _BaseRNG, arms: List[Arm], n_jobs: int, backend: Optional[str], mab_policy: Callable):
        super().__init__(rng, arms, n_jobs, backend)
        self.mab_policy = mab_policy

        self.arm_to_tree = dict.fromkeys(self.arms, DecisionTreeClassifier())
        # for each arm, keep dict of leaves and their reward list(ndarray)
        self.arm_to_rewards = dict.fromkeys(self.arms, dict())

    def fit(self, decisions: np.ndarray, rewards: np.ndarray, contexts: np.ndarray = None) -> NoReturn:

        # Reset the decision trees of each arm
        reset(self.arm_to_tree, DecisionTreeClassifier())
        reset(self.arm_to_rewards, dict())

        # Calculate fit
        self._parallel_fit(decisions, rewards, contexts)

    def partial_fit(self, decisions: np.ndarray, rewards: np.ndarray, contexts: np.ndarray = None) -> NoReturn:

        # Update rewards list at leaf for each arm
        for arm in self.arms:
            arm_contexts = contexts[decisions == arm]
            arm_rewards = rewards[decisions == arm]
            leaf_indices = self.arm_to_tree[arm].apply(arm_contexts)
            for i, leaf_index in enumerate(leaf_indices):
                leaf_rewards = self.arm_to_rewards[arm][leaf_index]
                self.arm_to_rewards[arm][leaf_index] = np.append(leaf_rewards, arm_rewards[i])

        # this algorithm won't do actual partial fit, so we won't call parallel_fit/fit_arm again
        # Calculate fit
        # self._parallel_fit(decisions, rewards, contexts)

    def predict(self, contexts: np.ndarray = None) -> Arm:

        # Return the first arm with maximum expectation
        return argmax(self.arm_to_expectation)

    def predict_expectations(self, contexts: np.ndarray = None) -> Dict[Arm, Num]:

        # For each arm, follow the contexts in each arm's tree to find leaf
        # TODO: can this work with more than one context at a time?
        # TODO: can this work without context?
        for arm in self.arms:
            # Assuming there is only one context, find leaf for that context
            leaf_index = self.arm_to_tree.apply(contexts[0])[0]

            # Go to the reward list for that leaf
            leaf_rewards = self.arm_to_rewards[arm][leaf_index]

            # Apply mab_policy to update arm_to_expectation
            # TODO: (just picked a random reward for now)
            self.arm_to_expectations[arm] = self.rng.choice(leaf_rewards)

        return self.arm_to_expectation.copy()

    def _fit_arm(self, arm: Arm, decisions: np.ndarray, rewards: np.ndarray, contexts: Optional[np.ndarray] = None):

        # Create dataset for the given arm
        # TODO: is it possible to implement this (fit decision tree) without context?
        arm_contexts = contexts[decisions == arm]
        arm_rewards = rewards[decisions == arm]

        # Fit decision tree to this dataset (update arm_to_tree for this arm)
        self.arm_to_tree[arm].fit(arm_contexts, arm_rewards)

        # For each leaf, keep a list of rewards
        leaf_indices = self.arm_to_tree[arm].apply(arm_contexts)

        # TODO: go through leaf indices and update each leaf's rewards list

    def _predict_contexts(self, contexts: np.ndarray, is_predict: bool,
                          seeds: Optional[np.ndarray] = None, start_index: Optional[int] = None) -> List:
        pass

    def _uptake_new_arm(self, arm: Arm, binarizer: Callable = None, scaler: Callable = None):
        self.arm_to_tree[arm] = DecisionTreeClassifier()
        self.arm_to_rewards[arm] = dict()