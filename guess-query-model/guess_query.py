import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
from scipy.optimize import curve_fit

class GuessQueryProblem:

  def __init__(self, 
              discount_factor, 
              guess_correct_reward,
              guess_wrong_cost,
              query_cost, 
              N, 
              convergence_tol
  ):
    self.discount_factor = discount_factor
    self.guess_correct_reward = guess_correct_reward
    self.guess_wrong_cost = guess_wrong_cost
    self.query_cost = query_cost
    self.N = N
    self.convergence_tol = convergence_tol
    self.discretization = None
    self.expected_value_function = None
    self.action_function = None
    self._optimal_actions = None
    self._scale = None
    self._squash = None

    self.QUERY = 0
    self.GUESS = 1

  def solve(self):
    self.discretization = np.linspace(0, 1, self.N)
    self.expected_value_function = np.zeros(self.N)

    # action, m_index state, x_index arrival
    self.action_function = np.zeros((2, self.N, self.N))
    for m_index in tqdm_notebook(range(self.N)):
      self.solve_step(m_index)

  def solve_step(self, m_index):
    abs_error = 1
    while abs_error > self.convergence_tol:
      value = self.average_max_value(m_index)
      abs_error = abs(value - self.expected_value_function[m_index])
      self.expected_value_function[m_index] = value
    
    # on final iteration, want to see what the calculation was
    # if m_index == 50:
    #   self.average_max_value(m_index, print_info=True)

  def guess_value(self, m_index, x_index):
    m = self.discretization[m_index]
    x = self.discretization[x_index]
    reward = None
    if x >= m:
      reward = self.guess_correct_reward
    else:
      prob_right_if_x_unknown = max(x, m - x) / m
      reward = self.guess_correct_reward if x >= m else self.guess_correct_reward * prob_right_if_x_unknown - self.guess_wrong_cost * (1 - prob_right_if_x_unknown)
    # If x is not in the unknown region (points less than m), we guess correctly.
    # Otherwise, we get it right with probability equal to the ratio of the bigger half of the unknown region to the whole unknown region.

    # Wow, I'm an idiot! Look at this bullshit. I can't believe I did this.
    # If you guess, you get this recursion.
    # How to handle this? Do a bunch of iterations? You just need to unroll this until the value of expected_value_function[m_index] drops out.
    # The value function is going to be the solution to an equation.
    # However, we don't know the coefficient of the expected_value_function[m_index]
    # because it's inside the max. 
    # If I keep iterating this, it should converge, right?
    # Why would it? if expected_value_function starts at 0, it goes to some higher value, which forces a higher value-function again... it should increase.
    return reward + self.discount_factor * self.expected_value_function[m_index]
    # The value of m (and thus m_index) does not change.

  def query_value(self, m_index, x_index):
    m = self.discretization[m_index]
    x = self.discretization[x_index]
    if x >= m:
      # No sane strategy queries in this case--it learns nothing and forgoes reward 1. 
      return self.discount_factor * self.expected_value_function[m_index] - self.query_cost
    else:
      # With probability x, we reduce the unknown region to x, and with probability m - x, we reduce the unknown region to m - x.
      assert x_index < m_index 
      return -self.query_cost + self.discount_factor * ((x/m) * self.expected_value_function[x_index] + ((m - x)/m) * self.expected_value_function[m_index - x_index])

  def average_max_value(self, m_index, print_info=False):
    # for print_info
    opt = self.GUESS

    value = 0
    for x_index in range(self.N):
      q = self.query_value(m_index, x_index)
      self.action_function[0, m_index, x_index] = q
      g = self.guess_value(m_index, x_index)
      self.action_function[1, m_index, x_index] = g
      value_of_opt_action = max(q, g)
      
      if print_info:
        print("index: " + str(x_index))
        if q > g:
          if opt == self.QUERY:
            print("index")
            print("OPT = query")
        else:
          print("OPT = guess")
      # if m_index == 0:
      #   print(value_of_opt_action)
      value += (1 / self.N) * value_of_opt_action
    return value 

  def optimal_actions(self):
    if self._optimal_actions is None:
      self._optimal_actions = np.argmax(self.action_function, axis = 0)
    return self._optimal_actions

  def optimal_actions_at_state_index(self, m_index):
    return self.optimal_actions()[m_index]

  def start_querying_indices(self):
    return np.argmax(self.optimal_actions() == 0, axis = 1)

  def always_guess_index(self):
    return np.argmax(self.start_querying_indices() > 0)

  def always_guess_point(self):
    return self.p_of_i(self.always_guess_index())

  def plot(self, curve, a_slice):
    plt.plot(self.discretization[a_slice], curve[a_slice])

  def p_of_i(self, index):
    return self.discretization[index]

  def omniscient_value(self):
    return 1 / (1 - self.discount_factor)

  def always_guess_index_value(self):
    return 1 / (1 - self.discount_factor) * \
        (1 - self.p_of_i(self.always_guess_index())/4)

  def scale(self):
    if self._scale is None:
      self.fit_log_section_of_ev_function()
    return self._scale 

  def squash(self):
    if self._squash is None:
      self.fit_log_section_of_ev_function()
    return self._squash

  def fit_log_section_of_ev_function(self):
    agi = self.always_guess_index()
    agp = self.always_guess_point()
    agv = self.always_guess_index_value()
    print(agi)
    print(agp)
    print(agv)
    fit_function = lambda m, scale, squash: agv - \
        scale * np.log(1 + squash * (m - agp))

    popt, pcov = curve_fit(fit_function, 
                           self.discretization[agi:],
                           self.expected_value_function[agi:])

    scale, squash = popt[0], popt[1]

    error = np.max(self.expected_value_function[agi:] - \
        fit_function(self.discretization[agi:], *popt))
    print("error: " + str(error))
    assert error < 0.01

    self._scale = scale
    self._squash = squash
    





    



  
  

  

    