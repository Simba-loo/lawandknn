import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors
import pdb
from collections import namedtuple
import math


HistoryEntry = namedtuple("HistoryEntry", ["case", "decision", "set_precedent"])

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
	knn = (all_cases[i] for i in indices)
	k_distances = distances[0:k]
	return(knn,k_distances)

def uniform_sample(d, l, r):
	# samples case uniformly from d-dimentional cube
	x = np.zeros(d)
	for i in range(d):
		x[i] = l+(r-l)*np.random.uniform()
	return(x)


# decision rules: output the decision and whether to set precedent
def distance_limited_precedence(x, judge_distribution, precedents, outcomes, k, max_distance, knn_tree):
	# if there are k precedents within max distance, settle case without setting precedent
	# if there aren't, settle according to judge and 
	if len(precedents)>k:
		knn, k_distances = query_knn_tree(knn_tree, precedents, x, k)
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
		k_decisions = [outcomes[tuple(e)] for e in knn]
		# majority rule
		decision = sum(k_decisions)> k/2.0
	return(decision, set_precedent)


# p(x) functions (fraction of judges thinking "yes" for case x)
def constant_func(x,c):
	return c


def scatter_cases(history):
	cases = [e[0] for e in history]
	plt.scatter([e[0] for e in cases],[e[1] for e in cases])
	plt.show()


def scatter_cases_with_outcomes(history):
	# positive_cases = [e[0] for e in history if e[1]]
	# negative_cases = [e[0] for e in history if not e[1]]
	colors = ["green" if e.decision else "red" for e in history]
	edgecolors = ["black" if e.set_precedent else "none" for e in history]
	plt.scatter([e.case[0] for e in history], [e.case[1] for e in history], 
		c = colors, edgecolors=edgecolors)
	# plt.scatter([c[0] for c in positive_cases],[c[1] for c in positive_cases],
	# 	c = "green")
	# plt.scatter([c[0] for c in negative_cases],[c[1] for c in negative_cases],
	# 	c = "red")
	plt.show()


def loss(history, judge_distribution):
	# loss is measured by the number of incorrect decisions
	# probably a better alternative is to incorporate what fraction of judges actually disagree to a given decision
	total_loss = 0
	for h in history:
		x = h[0]
		decision = h[1]
		ground_truth = judge_distribution(x)>0.5
		if decision!=ground_truth:
			total_loss+=1
	average_loss = total_loss*1.0/len(history)
	return(average_loss)

def run(decision_rule, k, judge_distribution, case_sampling_func, N):
	"""
	Run a simulated legal system.

	At each time step, a case c in D is randomly drawn from a fixed distribution Pr_D(.). It is judged (assigned a boolean label) according to a decision rule R. R either sets precedent (outputs true with probability Pr_J(true|c) for some fixed distribution Pr_J(., .))
	or does not (labels c somehow based on previous decisions). 
	Maintains history (list) consisting of tuples (c, decision, set_precedent),
		where c is a case, decision is its label, and set_precedent is True iff the decision rule set a precedent when judging c.

	Returns: history after judging N cases. 

	Parameters:
	N (int): Number of cases to run.
	judge_distribution (Map D -> [0, 1]): judge_distribution(c) = Pr_J(1|c)
	case_sampling_func (Map () -> D): case_sampling_func() is a sample from Pr_D.
	decision_rule (Map D -> Pr_J(1|.) -> precedents -> outcomes -> {0, 1} * bool):
		Represents the decision rule R.
	"""
	
	# Cache of the precedent cases and their outcomes (redundant with history)
	# used for efficient decisions
	precedent_cases = []
	precedent_outcomes = {}
	knn_tree = None

	# run the specified decision process for N
	history = []
	for _ in range(N):
		# sample a case 
		x = case_sampling_func()
		# decide it
		decision, set_precedent = decision_rule(x, judge_distribution, precedent_cases, precedent_outcomes, knn_tree)
		# update the caches
		if set_precedent:
			precedent_cases.append(x)
			precedent_outcomes[tuple(x)] = decision
			# fix this!
			knn_tree = build_knn_tree(precedent_cases, 7)
		# update the history
		# history.append((x,decision, set_precedent))
		history.append(HistoryEntry(x, decision, set_precedent))
	return history


d = 2 # dimension
l = 0 # left border
r = 1 # right border
k = 7 # make it odd to avoid draws
# distance within which to look for precedents
# max_distance = 0.05 
max_distance = 0.05
judge_distribution = lambda x: constant_func(x,0.9)
N = 10000

volume = (l - r) ** d
density = N / volume
volume_of_maxdist_ball =  math.pi * max_distance ** 2
number_in_maxdist_ball = density * volume_of_maxdist_ball
N_at_which_precedent_is_as_likely_as_not = k * volume / volume_of_maxdist_ball

print("volume = " + str(volume))
print("density = " + str(density))
print("volume searched for neighbors = " + str(volume_of_maxdist_ball))
print("expected number of neighbors of last case judged = " + str(number_in_maxdist_ball))
print("N where density favors judging via precedent = " + 
	str(N_at_which_precedent_is_as_likely_as_not))


history = run(N=N,
							k = k,
							judge_distribution=judge_distribution,
							case_sampling_func = lambda: uniform_sample(d, l, r),
							decision_rule = lambda x, judge_distribution, precedents, 						outcomes, knn_tree: distance_limited_precedence(x, judge_distribution, 	precedents, outcomes, k, max_distance, knn_tree)
)

loss = loss(history, judge_distribution)

h_outcomes = [e.decision for e in history] 
h_set_precedent = [e.set_precedent for e in history]

print("Average loss per case:")
print(loss)
print("fraction of positive outcomes:")
print(sum(h_outcomes)/N)
print("precedents to all cases ratio")
print(sum(h_set_precedent)/N)

# scatter_cases(history)
scatter_cases_with_outcomes(history)

# when are precedents established? My guess is system exhibits phase transition
# Not as sharp as I expected for small N, but there is a phase transition
# near N_at_which_precedent_is_as_likely_as_not
# Are real common law systems "saturated" by precedent?
# Is this problem alleviated by decaying precedents?
precedent_indicators = [1 if e.set_precedent else 0 for e in history]
precedents_set_up_to_t = np.cumsum(np.asarray(precedent_indicators))
plt.plot(precedents_set_up_to_t)
plt.show()
