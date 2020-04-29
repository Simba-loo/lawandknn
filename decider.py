"""For decision rules that maintain state"""
from abc import ABC
from sklearn.neighbors import NearestNeighbors
import numpy as np


class Decider(ABC):
  """Decider represents a decision rule that maintains its own history and state"""

  def decide(self, x, judge_distribution):
    decision, set_precedent = self.apply_rule(x, judge_distribution)
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
      self.knn_tree = build_knn_tree(self.precedents, self.k)


class DistanceLimitedForgetfulDecider(DistanceLimitedDecider):
  
  def __init__(self, k, max_distance, horizon):
    super().__init__(k, max_distance)
    self.horizon = horizon

  def update(self, x, decision, set_precedent):
    if set_precedent and len(self.precedents) > self.horizon:
      self.precedents = self.precedents[-self.horizon:]
      self.outcomes = self.outcomes[-self.horizon:]
    super().update(x, decision, set_precedent)



def find_knn(all_cases, target_case, k):
	# returns nearest neighbors and distances
	# note: this entire tree is constructed every time a case is judged.
	# This is probably the simulation bottleneck.
	nbrs = build_knn_tree(all_cases, k)
	return query_knn_tree(nbrs, all_cases, target_case, k)

def build_knn_tree(all_cases, k):
	return NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(all_cases)

def query_knn_tree(knn_tree, all_cases, target_case, k):
	distances, indices = knn_tree.kneighbors([target_case])
	distances = distances[0]
	indices = indices[0]
	#pdb.set_trace()
	# knn = (all_cases[i] for i in indices)
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
		# set a new precedent
		# precedents.append(x)
		decision = np.random.uniform()<judge_distribution(x)
		# x_tuple = tuple(x)
		# outcomes[x_tuple] = decision
	else:
		# get decisions from nearest precedents
		k_decisions = [outcomes[index] for index in indices]
		# majority rule
		decision = sum(k_decisions)> k/2.0
	return(decision, set_precedent)