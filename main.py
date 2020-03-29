import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors
import pdb

# set the domain


def run(decision_rule, judge_distribution,case_sampling_func, precedent_cases, precedent_outcomes, N):
	# run the specified decition process for N
	history = []
	for i in range(N):
		# sample a case 
		x = case_sampling_func()
		decision, set_precedent, precedents, outcomes = decision_rule(x, judge_distribution, precedent_cases, precedent_outcomes)
		history.append((x,decision, set_precedent))
	return(precedent_cases, precedent_outcomes, history)




def find_knn(all_cases, target_case, k):
	# returns nearest neighbors and distances
	nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(all_cases)
	distances, indices = nbrs.kneighbors([target_case])
	distances = distances[0]
	indices = indices[0]
	#pdb.set_trace()
	knn = (all_cases[i] for i in indices)
	k_distances = distances[0:k]
	return(knn,k_distances)


def uniform_sample(d, l,r):
	# samples case uniformly from d-dimentional cube
	x = np.zeros(d)
	for i in range(d):
		x[i] = l+(r-l)*np.random.uniform()
	return(x)



# decision rules: output the decision and whether to set precedent
def distance_limited_precedence(x, distribution, precedents, outcomes, k, max_distance):
	# if there are k precedents within max distance, settle case without setting precedent
	# if there aren't, settle according to judge and 
	if len(precedents)>k:
		knn, k_distances = find_knn(precedents, x, k)
		set_precedent = not np.all(k_distances<max_distance)
	else: 
		set_precedent = True
	if set_precedent:
		# set a new precedent
		precedents.append(x)
		decision = np.random.uniform()<distribution(x)
		x_tuple = tuple(x)
		outcomes[x_tuple] = decision
	else:
		# get decisions from nearest precedents
		k_decisions = [outcomes[tuple(e)] for e in knn]
		# majority rule
		decision = sum(k_decisions)> k/2.0
	return(decision, set_precedent, precedents, outcomes)

# p(x) functions (fraction of judges thinking "yes" for case x)
def constant_func(x,c):
	return c

def scatter_cases(history):
	cases = [e[0] for e in history]
	plt.scatter([e[0] for e in cases],[e[1] for e in cases])
	plt.show()

def scatter_cases_with_outcomes(history):
	positive_cases = [e[0] for e in history if e[2]]
	negative_cases = [e[0] for e in history if not e[2]]
	plt.scatter([e[0] for e in positive_cases],[e[1] for e in positive_cases])
	plt.scatter([e[0] for e in negative_cases],[e[1] for e in negative_cases])
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


# set domain
d = 2 # dimension
l = 0 # left border
r = 1 # right border
# set the judge opinion distribution
judge_distribution = lambda x: constant_func(x,0.7)
# how to sample cases from domain
case_sampling_func = lambda: uniform_sample(d,l,r)
# set decision rule
k=7 # make it odd to avoid draws
max_distance = 0.05 # distance within which to look for precedents
decision_rule = lambda x, judge_distribution, precedents, outcomes: distance_limited_precedence(x, judge_distribution, precedents, outcomes, k, max_distance)

# initialize precedence data
precedent_cases = []
precedent_outcomes = {} # case -> outcome dictinary

# number of cases to run
N = 10000

precedent_cases, precedent_outcomes, history = run(decision_rule, judge_distribution,case_sampling_func, precedent_cases, precedent_outcomes, N)
h_outcomes = [e[1] for e in history] 
h_set_precedent = [e[2] for e in history]

# scatter_cases(history)
scatter_cases_with_outcomes(history)
loss = loss(history, judge_distribution)

print("Average loss per case:")
print(loss)
print("fraction of positive outcomes:")
print(sum(h_outcomes)/N)
print("precedents to all cases ratio")
print(sum(h_set_precedent)/N)