import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors
import pdb
from collections import namedtuple
import math
from decider import DistanceLimitedDecider, DistanceLimitedForgetfulDecider


HistoryEntry = namedtuple("HistoryEntry", ["case", "decision", "set_precedent"])

HistoryEntryWithLoss = namedtuple("HistoryEntryWithLoss", ["case", "decision", "set_precedent", "loss", "nonprecedent_loss"])



def uniform_sample(d, l, r):
	# samples case uniformly from d-dimentional cube
	x = np.zeros(d)
	for i in range(d):
		x[i] = l+(r-l)*np.random.uniform()
	return(x)


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
	# edgecolors = ["black" if e.set_precedent == True else ("blue" if e.set_precedent == 2 else "none") for e in history]
	edgecolors = ["black" if e.set_precedent == True else "none" for e in history]
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
	average_loss = total_loss*1.0/len(history) if len(history) > 0 else "undefined"
	return(average_loss)

def run(decider, judge_distribution, case_sampling_func, N):
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

	# run the specified decision process for N
	history = []
	loss = 0
	nonprecedent_loss = 0
	for _ in range(N):
		# sample a case 
		x = case_sampling_func()
		# decide it
		decision, set_precedent = decider.decide(x, judge_distribution)
		ground_truth = judge_distribution(x) > 0.5
		if decision != ground_truth:
			loss += 1
			if not set_precedent:
				nonprecedent_loss += 1
		# update the history
		# history.append((x,decision, set_precedent))
		history.append(
			HistoryEntryWithLoss(
				case=x, 
				decision=decision, 
				set_precedent=set_precedent, 
				loss=loss, 
				nonprecedent_loss=nonprecedent_loss))
	return history

