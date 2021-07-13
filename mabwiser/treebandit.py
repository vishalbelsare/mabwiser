# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from typing import Dict, List, NoReturn, Optional, Callable

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from mabwiser.base_mab import BaseMAB
from mabwiser.utils import reset, argmax, Arm, Num, _BaseRNG


class _TreeBandit(BaseMAB):
    # TODO: set default for mab_policy
    def __init__(self, rng: _BaseRNG, arms: List[Arm], n_jobs: int, backend: Optional[str],
                 mab_policy: Optional[Callable] = None):
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

        return self._parallel_predict(contexts, is_predict=True)

    def predict_expectations(self, contexts: np.ndarray = None) -> Dict[Arm, Num]:

        return self._parallel_predict(contexts, is_predict=False)

    def _fit_arm(self, arm: Arm, decisions: np.ndarray, rewards: np.ndarray, contexts: Optional[np.ndarray] = None):
        # Create dataset for the given arm
        arm_contexts = contexts[decisions == arm]
        arm_rewards = rewards[decisions == arm]

        # Fit decision tree to this dataset (update arm_to_tree for this arm)
        self.arm_to_tree[arm].fit(arm_contexts, arm_rewards)

        # For each leaf, keep a list of rewards
        context_leaf_indices = self.arm_to_tree[arm].apply(arm_contexts)
        leaf_indices = set(context_leaf_indices)

        for index in leaf_indices:
            # Get rewards list for each leaf
            self.arm_to_rewards[arm][index] = arm_rewards[context_leaf_indices == index]

    def _predict_contexts(self, contexts: np.ndarray, is_predict: bool,
                          seeds: Optional[np.ndarray] = None, start_index: Optional[int] = None) -> List:

        # Get local copy of model, arm_to_expectation and arms to minimize
        # communication overhead between arms (processes) using shared objects
        arm_to_tree = deepcopy(self.arm_to_tree)
        arm_to_rewards = deepcopy(self.arm_to_rewards)
        arm_to_expectation = deepcopy(self.arm_to_expectation)
        arms = deepcopy(self.arms)

        # Create an empty list of predictions
        predictions = [None] * len(contexts)
        for index, row in enumerate(contexts):
            for arm in arms:
                # Go to that arm's tree, follow context, reach leaf
                leaf_index = arm_to_tree[arm].apply([row])[0]

                # Find expected reward
                leaf_rewards = arm_to_rewards[arm][leaf_index]

                # find expectation, update arm_to_expectation for that arm
                # Apply mab_policy to update arm_to_expectation
                # TODO: (just picked random value for now)
                arm_to_expectation[arm] = self.rng.choice(leaf_rewards)

            if is_predict:
                predictions[index] = argmax(arm_to_expectation)
            else:
                predictions[index] = arm_to_expectation.copy()

        # Return list of predictions
        return predictions

    def _uptake_new_arm(self, arm: Arm, binarizer: Callable = None, scaler: Callable = None):
        self.arm_to_tree[arm] = DecisionTreeClassifier()
        self.arm_to_rewards[arm] = dict()
