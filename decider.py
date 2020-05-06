"""For decision rules that maintain state"""
from abc import ABC
from sklearn.neighbors import NearestNeighbors
import numpy as np


class Decider(ABC):
  """Decider represents a decision rule that maintains its own history and state"""

  def decide(self, x, judge_distribution):
    # Separation of responsibilities. First make a decision based on current history / state.
    decision, set_precedent = self.apply_rule(x, judge_distribution)
    # Then update state to prepare for deciding next case.
    self.update(x, decision, set_precedent)
    return decision, set_precedent

  def apply_rule(self, x, judge_distribution):
    pass

  def update(self, x, decision, set_precedent):
    pass


class DistanceLimitedDecider(Decider):

  def __init__(self, k, max_distance):
    self.k = k
    self.max_distance = max_distance
    self.precedents = []
    self.outcomes = []
    self.knn_tree = None

  def apply_rule(self, x, judge_distribution):
    decision, set_precedent = distance_limited_precedence(x, judge_distribution, self.precedents, self.outcomes, self.k, self.max_distance, self.knn_tree)
    return decision, set_precedent

  def update(self, x, decision, set_precedent):
    if set_precedent:
      self.precedents.append(x)
      self.outcomes.append(decision)
      # Rebuild the knn tree only after setting a precedent.
      self.knn_tree = build_knn_tree(self.precedents, self.k)


class DistanceLimitedForgetfulDecider(DistanceLimitedDecider):
  
  def __init__(self, k, max_distance, horizon):
    super().__init__(k, max_distance)
    self.horizon = horizon

  def update(self, x, decision, set_precedent):
    # Forget precedents set longer than horizon ago.
    if set_precedent and len(self.precedents) > self.horizon:
      self.precedents = self.precedents[-self.horizon:]
      self.outcomes = self.outcomes[-self.horizon:]
    # Add the latest precedent and rebuild the knn tree.
    super().update(x, decision, set_precedent)

class DistanceLimitedTimedDecider(DistanceLimitedDecider):
  def __init__(self, k, max_distance):
    super().__init__(k, max_distance)
    self.current_time = 0
    self.timestamps = []

  def update(self, x, decision, set_precedent):
    self.current_time += 1
    super().update(x, decision, set_precedent)
    if set_precedent:
      self.timestamps.append(self.current_time)

class DistanceLimitedDropoutDecider(DistanceLimitedTimedDecider):
  def __init__(self, k, max_distance, half_life, dropout_interval):
    super().__init__(k, max_distance)
    self.half_life = half_life
    self.dropout_interval = dropout_interval

  def update(self, x, decision, set_precedent):
    if self.current_time % self.dropout_interval == 0:
      surviving_indices = self.get_surviving_indices()
      self.precedents = self.select_surviving(self.precedents,        surviving_indices)
      self.outcomes = self.select_surviving(self.outcomes, surviving_indices)
      self.timestamps = self.select_surviving(self.timestamps, surviving_indices)
      if len(self.precedents) > 0:
        self.knn_tree = build_knn_tree(self.precedents, self.k)
    super().update(x, decision, set_precedent)
  
  def get_surviving_indices(self):
    np_timestamps = np.asarray(self.timestamps)
    # lifetimes = self.current_time - np_timestamps
    decay_probabilities = np.ones(np_timestamps.shape) - np.exp2(-self.dropout_interval / self.half_life)
    surviving_indices = np.random.uniform(size = np_timestamps.shape) >  decay_probabilities
    return surviving_indices

  @staticmethod
  def select_surviving(lst, surviving_indices):
    np_lst = np.asarray(lst)
    np_select = np_lst[surviving_indices]
    return list(np_select)



def find_knn(all_cases, target_case, k):
	# returns nearest neighbors and distances
	# note: this entire tree is constructed every time a case is judged.
	# This is probably the simulation bottleneck.
  # Would be better to add new cases as they are judged, but sklearn
  # does not seem to support this.
	nbrs = build_knn_tree(all_cases, k)
	return query_knn_tree(nbrs, all_cases, target_case, k)

def build_knn_tree(all_cases, k):
	return NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(all_cases)

def query_knn_tree(knn_tree, all_cases, target_case, k):
	distances, indices = knn_tree.kneighbors([target_case])
	distances = distances[0]
	indices = indices[0]
	k_distances = distances[0:k]
	return(indices,k_distances)

# decision rules: output the decision and whether to set precedent
def distance_limited_precedence(x, judge_distribution, precedents, outcomes, k, max_distance, knn_tree):
	# if there are k precedents within max distance, settle case without setting precedent
	# if there aren't, settle according to judge and 
	if len(precedents)>k:
		indices, k_distances = query_knn_tree(knn_tree, precedents, x, k)
		set_precedent = not np.all(k_distances<max_distance)
	else: 
		set_precedent = True
	if set_precedent:
		decision = np.random.uniform()<judge_distribution(x)
	else:
		# get decisions from nearest precedents
		k_decisions = [outcomes[index] for index in indices]
		# majority rule
		decision = sum(k_decisions)> k/2.0
	return(decision, set_precedent)