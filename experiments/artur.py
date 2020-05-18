import numpy as np
import math 
from matplotlib import pyplot as plt
import sys
sys.path.append("..")
from simulation import run, constant_func, scatter_cases_with_outcomes
from simulation import uniform_sample, loss
from decider import DistanceLimitedDecider, DistanceLimitedForgetfulDecider
from decider import SuperPrecedentsDecider
import pdb

def main():
  d = 2 # dimension
  l = 0 # left border
  r = 1 # right border
  k = 7 # make it odd to avoid draws
  # distance within which to look for precedents
  # max_distance = 0.05 
  max_distance = 0.05
  judge_distribution = lambda x: constant_func(x,0.7)
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
                judge_distribution=judge_distribution,
                case_sampling_func = lambda: uniform_sample(d, l, r),
                # decider = DistanceLimitedDecider(k, max_distance)
                # decider = DistanceLimitedDecider(k, max_distance, 1000)
                decider = SuperPrecedentsDecider([k,k], [max_distance,3*max_distance],)

  )

  experiment_loss = loss(history, judge_distribution)

  h_outcomes = [e.decision for e in history] 
  h_set_precedent = [e.set_precedent for e in history]

  print("Average loss per case:")
  print(experiment_loss)
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
  precedent_indicators = [1 if e.set_precedent==True else 0 for e in history]
  precedents_set_up_to_t = np.cumsum(np.asarray(precedent_indicators))
  plt.plot(precedents_set_up_to_t)
  plt.show()

  # this is for super precedents, feel free to comment out
  super_precedent_indicators = [1 if e.set_precedent==False else 0 for e in history] # False means there wasn't enough super-precedents but enough precedents to settle a case, so a super precedent was added
  super_precedents_set_up_to_t = np.cumsum(np.asarray(super_precedent_indicators))
  plt.plot(super_precedents_set_up_to_t)
  plt.show()

if __name__ == "__main__":
  main()