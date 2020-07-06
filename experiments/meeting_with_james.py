# Any ideas for good simulations?
# Maybe just a basic contrast of distance limited decider vs. super precedents decider vs. threshold majority... no, that's too complicated.
# Pick the same set of parameters (linear boundary?). Show one example of distance limited decider as an example of informational herding, plus all the other variants just for fun.

import numpy as np
import math 
from matplotlib import pyplot as plt
import sys
sys.path.append("..")
from simulation import run, constant_func, scatter_cases_with_outcomes
from simulation import uniform_sample, loss
from decider import DistanceLimitedDecider, DistanceLimitedForgetfulDecider
from decider import DistanceLimitedDropoutDecider, CaseByCaseDecider
from decider import SuperPrecedentsDecider, DistanceLimitedHarmonicOverrulingDecider, DistanceLimitedConstantOverrulingDecider, DistanceLimitedTimedDecider, DistanceLimitedThresholdMajorityDecider  
from collections import namedtuple
from statistics import mean

Setup2D = namedtuple("Setup2D", ["l", "r", "d", "N", "max_distance", "k", "judge_distribution", "case_sampling_func", "make_decider"])

def make_setups():
  # Setup
  d = 2 # dimension
  l = 0 # left border
  r = 1 # right border
  k = 7
  max_distance = 0.1
  judge_distribution = lambda x: 0.7 if x[0] < 0.5 else 0.3
  N = 1000
  case_sampling_func = lambda: uniform_sample(d, l, r)
  make_deciders = [
    lambda: CaseByCaseDecider(),
    lambda: DistanceLimitedDecider(
      k_of_self=k,
      max_distance_of_self=max_distance),
    lambda: DistanceLimitedConstantOverrulingDecider(
      k_of_self=k, 
      max_distance_of_self=max_distance,
      overruling_probability=0.1),
    lambda: DistanceLimitedDropoutDecider(k, max_distance, 
      half_life = 10000000,
      dropout_interval = 100),
    lambda: SuperPrecedentsDecider([k,k], [max_distance, 3*max_distance]),
    lambda: DistanceLimitedThresholdMajorityDecider(
      k_of_self = k,
      max_distance_of_self = max_distance,
      threshold_of_self = 1
    ),
    # DistanceLimitedTimedDecider(
    #   k_of_self = lambda self: math.floor(math.log(max(self.current_time, math.exp(7)))),
    #   max_distance_of_self = max_distance,
    #   hreshold_of_self = lambda self: math.sqrt(self.k_of_self(self))/4
    # )
    #  DistanceLimitedThresholdMajorityDecider(
    #   k_of_self = lambda self: math.floor(math.log(max(self.current_time,     math.exp(7)))),
    #   max_distance_of_self = max_distance,
    #   threshold_of_self = lambda self: math.sqrt(self.k_of_self(self))/4
    # )
  ]

  setups = [Setup2D(l=l, r=r, d=d, N=N, max_distance=max_distance, k=k, judge_distribution=judge_distribution, case_sampling_func=case_sampling_func, make_decider = make_decider) for make_decider in make_deciders]
  return setups


def analyze(s):
  # Printouts
  if type(s.k) == int and type(s.max_distance) == float:
    volume = (s.l - s.r) ** s.d
    density = s.N / volume
    volume_of_maxdist_ball =  math.pi * s.max_distance ** 2
    number_in_maxdist_ball = density * volume_of_maxdist_ball
    N_at_which_precedent_is_as_likely_as_not = s.k * volume / volume_of_maxdist_ball
    print("volume = " + str(volume))
    print("density = " + str(density))
    print("volume searched for neighbors = " + str(volume_of_maxdist_ball))
    print("expected number of neighbors of last case judged = " + str(number_in_maxdist_ball))
    print("N where density favors judging via precedent = " + 
      str(N_at_which_precedent_is_as_likely_as_not))

  history = run(N=s.N,
                judge_distribution=s.judge_distribution,
                case_sampling_func=s.case_sampling_func,
                decider = s.make_decider(),
  )

  # Summary
  experiment_loss = loss(history, s.judge_distribution)
  h_outcomes = [e.decision for e in history] 
  h_derived = [e for e in history if not e.set_precedent]
  derived_loss = loss(h_derived, s.judge_distribution)
  h_set_precedent = [e.set_precedent for e in history]

  print("Average loss per case:")
  print(experiment_loss)
  print("Average loss of cases judged by precedent:")
  print(derived_loss)
  print("fraction of positive outcomes:")
  print(sum(h_outcomes)/s.N)
  print("precedents to all cases ratio")
  print(sum(h_set_precedent)/s.N)

  # Plots
  scatter_cases_with_outcomes(history)

  precedent_indicators = [1 if e.set_precedent else 0 for e in history]
  precedents_set_up_to_t = np.cumsum(np.asarray(precedent_indicators))
  plt.plot(precedents_set_up_to_t)
  plt.xlabel("t")
  plt.ylabel("precedents set up to t")
  plt.show()

  # nonprecedents_set_up_to_t = np.cumsum(np.logical_not(np.asarray(precedent_indicators)))
  # nonprec_loss_over_time = [e.nonprecedent_loss for e in history]
  # fractional_nonprec_loss_over_time = \
  #     np.asarray(nonprec_loss_over_time) / np.asarray(nonprecedents_set_up_to_t)
  # plt.plot(fractional_nonprec_loss_over_time)
  # plt.xlabel("t")
  # plt.ylabel("fractional nonprecedent loss up to t")
  # plt.show()

def analyze_averages(s, num_runs):
  losses = []
  derived_losses = []
  positive_to_all_cases_ratios = []
  precedent_to_all_cases_ratios = []
  for _ in range(num_runs):
    history = run(N=s.N,
                  judge_distribution=s.judge_distribution,
                  case_sampling_func=s.case_sampling_func,
                  decider = s.make_decider(),
    )

    losses.append(loss(history, s.judge_distribution))

    h_derived = [e for e in history if not e.set_precedent]
    derived_losses.append(loss(h_derived, s.judge_distribution))

    h_set_precedent = [e.set_precedent for e in history]
    precedent_to_all_cases_ratios.append(sum(h_set_precedent)/s.N)

    h_outcomes = [e.decision for e in history] 
    positive_to_all_cases_ratios.append(sum(h_outcomes)/s.N)

  try:
    average_loss = mean(losses)
    print("Loss: " + str(average_loss))
  except:
    print("Loss: undefined")

  try:
    derived_loss = mean(derived_losses)
    print("Loss of cases decided by precedent: " + str(derived_loss))
  except:
    print("Loss of cases decided by precedent: undefined")

  average_positive_ratio = mean(positive_to_all_cases_ratios)
  print("Fraction of positive cases: " + str(average_positive_ratio))

  # print(precedent_to_all_cases_ratios)
  average_precedent_ratio = mean(precedent_to_all_cases_ratios)
  print("Fraction of precedent cases: " + str(average_precedent_ratio))


def main():
  setups = make_setups()

  for s in setups:
    print(s.make_decider().__class__.__name__)
    analyze(s)
    print()

if __name__ == "__main__":
  main()