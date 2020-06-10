import numpy as np
import math 
from matplotlib import pyplot as plt
import sys
sys.path.append("..")
from simulation import run, constant_func, scatter_cases_with_outcomes
from simulation import uniform_sample, loss
from decider import DistanceLimitedDecider, DistanceLimitedForgetfulDecider
from decider import DistanceLimitedDropoutDecider, CaseByCaseDecider
from decider import SuperPrecedentsDecider, DistanceLimitedHarmonicOverrulingDecider, DistanceLimitedConstantOverrulingDecider, DistanceLimitedTimedDecider

def main():
  d = 2 # dimension
  l = 0 # left border
  r = 1 # right border
  k = 51 # make it odd to avoid draws
  # distance within which to look for precedents
  # max_distance = 0.05 
  max_distance = 0.1

  judge_distribution = lambda x: constant_func(x,0.7)
  # judge_distribution = lambda x: 0.5 + 0.5 * math.sin(2 * math.pi * x[0])
  # judge_distribution = lambda x: 0.5 + 0.5 * math.sin(4 * math.pi * x[0])
  # judge_distribution = lambda x: 0.5 + 0.5 * math.sin(8 * math.pi * x[0])
  # judge_distribution = lambda x: \
  #     0.5 + 0.5 * math.sin(8 * math.pi * x[0]) + \
  #           0.5 * math.sin(2 * math.pi * x[0])

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
                # decider = CaseByCaseDecider()
                # decider = DistanceLimitedDecider(k=k, max_distance=max_distance),
                # decider = DistanceLimitedConstantOverrulingDecider(k=k, max_distance=max_distance,
                # overruling_probability=0.1),  
                # decider = DistanceLimitedDropoutDecider(k, max_distance, 
                  # half_life = 10000000,
                  # dropout_interval = 100)
                # decider = SuperPrecedentsDecider([k,k], [max_distance,    3*max_distance],)
                decider = DistanceLimitedTimedDecider(
                    k_of_self = lambda self: math.floor(math.log(max(self.current_time, math.exp(7)))),
                    max_distance_of_self = max_distance
                )
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
  # plt.savefig("cases_with_outcomes.fig")

  # when are precedents established? My guess is system exhibits phase transition
  # Not as sharp as I expected for small N, but there is a phase transition
  # near N_at_which_precedent_is_as_likely_as_not
  # Are real common law systems "saturated" by precedent?
  # Is this problem alleviated by decaying precedents?
  precedent_indicators = [1 if e.set_precedent else 0 for e in history]
  precedents_set_up_to_t = np.cumsum(np.asarray(precedent_indicators))
  nonprecedents_set_up_to_t = np.cumsum(np.logical_not(np.asarray(precedent_indicators)))
  plt.plot(precedents_set_up_to_t)
  plt.xlabel("t")
  plt.ylabel("precedents set up to t")
  # plt.savefig("precedents_up_to_t.fig")
  plt.show()
  nonprec_loss_over_time = [e.nonprecedent_loss for e in history]
  fractional_nonprec_loss_over_time = \
      np.asarray(nonprec_loss_over_time) / np.asarray(nonprecedents_set_up_to_t)
  plt.plot(fractional_nonprec_loss_over_time)
  plt.xlabel("t")
  plt.ylabel("fractional nonprecedent loss up to t")
  # plt.savefig("loss_up_to_t.fig")
  plt.show()

if __name__ == "__main__":
  main()