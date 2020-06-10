"""For decision rules that maintain state"""
from abc import ABC, abstractmethod
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

  @abstractmethod
  def apply_rule(self, x, judge_distribution):
    pass

  @abstractmethod
  def update(self, x, decision, set_precedent):
    pass


class CaseByCaseDecider(Decider):

  def apply_rule(self, x, judge_distribution):
    return np.random.uniform() < judge_distribution(x), False


# class AbstractDistanceLimitedDecider(Decider):
#   @abstractmethod
#   def get_k(self):
#     pass 

#   @abstractmethod
#   def get_max_distance(self):
#     pass


class DistanceLimitedDecider(Decider):

  def __init__(self, k_of_self, max_distance_of_self):
    self.k_of_self = (lambda self: k_of_self) if type(k_of_self) == int else k_of_self
    self.max_distance_of_self = lambda self: max_distance_of_self if type(max_distance_of_self) == float else max_distance_of_self
    self.precedents = []
    self.outcomes = []
    self.knn_tree = None

  # def get_k(self):
  #   return self.k

  # def get_max_distance(self):
  #   return self.max_distance

  def apply_rule(self, x, judge_distribution):
    # print(self.k_of_self)
    # print(self.k_of_self(3))
    print(self.k_of_self(self))
    decision, set_precedent = distance_limited_precedence(x, judge_distribution, self.precedents, self.outcomes, self.k_of_self(self), self.max_distance_of_self(self), self.knn_tree)
    return decision, set_precedent

  def update(self, x, decision, set_precedent):
    if set_precedent:
      self.precedents.append(x)
      self.outcomes.append(decision)
      # Rebuild the knn tree only after setting a precedent.
      self.knn_tree = build_knn_tree(self.precedents, self.k_of_self(self))


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
  def __init__(self, k_of_self, max_distance_of_self):
    super().__init__(k_of_self, max_distance_of_self)
    self.current_time = 0
    self.timestamps = []

  def update(self, x, decision, set_precedent):
    self.current_time += 1
    super().update(x, decision, set_precedent)
    if set_precedent:
      self.timestamps.append(self.current_time)


class DistanceLimitedThresholdMajorityDecider(DistanceLimitedDecider):
  # TODO
  def __init__(self, k, max_distance_of_self, threshold):
    super().__init__(k, max_distance_of_self)
    self.threshold = threshold

  def apply_rule(self, x, judge_distribution):
    return distance_limited_precedence_with_threshold(x, judge_distribution,    self.precedents, self.outcomes, self.k_of_self(self), self.max_distance_of_self(self), self.knn_tree, self.threshold)


class DistanceLimitedOverrulingDecider(DistanceLimitedTimedDecider):

  @abstractmethod
  def probability_of_overruling(self):
    pass 

  def apply_rule(self, x, judge_distribution):
    if np.random.uniform() < self.probability_of_overruling():
      decision = np.random.uniform() < judge_distribution(x)
      return decision, True 
    else:
      return super().apply_rule(x, judge_distribution)


class DistanceLimitedHarmonicOverrulingDecider(DistanceLimitedOverrulingDecider):
  def probability_of_overruling(self):
    return 1/(self.current_time + 1)


class DistanceLimitedConstantOverrulingDecider(DistanceLimitedOverrulingDecider):
  def __init__(self, k, max_distance, overruling_probability):
    super().__init__(k, max_distance)
    self.overruling_probability = overruling_probability

  def probability_of_overruling(self):
    return self.overruling_probability


class DistanceLimitedDropoutDecider(DistanceLimitedTimedDecider):
  """
  Old precedents disappear randomly with a half life of half_life.
  """
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
        self.knn_tree = build_knn_tree(self.precedents, self.k_of_self(self))
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


class SuperPrecedentsDecider(Decider):

  def __init__(self, k, max_distance):
    self.k = k # this is now an tuple
    self.max_distance = max_distance # this is now an tuple
    self.precedents = [[],[]] # first list is precedents, second list is super precedents
    self.outcomes = [[],[]]
    self.knn_tree = [None,None]

  def apply_rule(self, x, judge_distribution):
    super_decision, set_super_precedent = distance_limited_precedence(x, judge_distribution, self.precedents[1], self.outcomes[1], self.k[1], self.max_distance[1], self.knn_tree[1])
    if set_super_precedent: # means not enough superprecedents around, check for regular precedents
      decision, set_precedent = distance_limited_precedence(x, judge_distribution, self.precedents[0], self.outcomes[0], self.k[0], self.max_distance[0], self.knn_tree[0])
    else:
      decision = super_decision
      set_precedent = 2 # mixing types here, sad. FYI (2==True) evaluates to False in python
    return decision, set_precedent

  def update(self, x, decision, set_precedent):
    # if it's a normal precedent
    if set_precedent==True:
      self.precedents[0].append(x)
      self.outcomes[0].append(decision)
      # Rebuild the knn tree only after setting a precedent.
      self.knn_tree[0] = build_knn_tree(self.precedents[0], self.k[0])
    # if it's settled by precedents, make it a super-precedent
    if set_precedent==False:
      self.precedents[1].append(x)
      self.outcomes[1].append(decision)
      # Rebuild the knn tree only after setting a precedent.
      self.knn_tree[1]= build_knn_tree(self.precedents[1], self.k[1])
    # if it's settled by superprecedents, do nothing


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
		decision = sum(k_decisions) > k/2.0
	return(decision, set_precedent)

def distance_limited_precedence_with_threshold(
    x, 
    judge_distribution,
    precedents,
    outcomes, 
    k,
    max_distance,
    knn_tree,
    threshold = lambda k: 0):
  judge_decision = np.random.uniform() < judge_distribution(x)
  if len(precedents) < k:
    return judge_decision, True
  else: 
    indices, k_distances = query_knn_tree(knn_tree, precedents, x, k)
    all_k_within_max_distance = np.all(k_distances<max_distance)
    if not all_k_within_max_distance:
      return judge_decision, True
    else:
      k_decisions = [outcomes[index] for index in indices]
      no_threshold_majority = abs(sum(k_decisions) - k/2.0) < threshold(k)
      if no_threshold_majority:
        return judge_decision, True
      else:
        return sum(k_decisions) > k/2.0, False
    
    