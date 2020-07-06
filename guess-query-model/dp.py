# Solving for the expected value function F_bar(m) as a function of the size of the unknown region, m, and the fixed discount factor beta, 

import numpy as np
import matplotlib.pyplot as plt

reward = 1  # assume query_cost = 0 and mistake_cost = 0 for now (A1)
discount_factor = 1 - 0.01
print("Discount factor: " + str(discount_factor))
assert(discount_factor > 0 and discount_factor < 1)

N = 1001
discretization = np.linspace(0, 1, N)  # values of m
print("Discretization: " + str(discretization))

value_function = np.zeros(N)

# m-value at which you should always guess, solved analytically under assn (A1)
# always_guess_point = (4 * (1 - discount_factor)) / discount_factor
# print("Always-guess point: " + str(always_guess_point))
# assert(always_guess_point > 0 and always_guess_point < 1)

# min_query_index = 0
# while True:
#   point = discretization[min_query_index]
#   if discretization[min_query_index] < always_guess_point:
#     value_function[min_query_index] = (1 - (1/8 * point)) / (1 - discount_factor)
#   else:
#     break
#   min_query_index = min_query_index + 1
# print("Min query index: " + str(min_query_index))
# print("Value function 1: " + str(value_function)


def guess_value(m_index, x_index):
  m = discretization[m_index]
  x = discretization[x_index]
  reward = 1 if x >= m else max(x, m - x) / m
  # If x is not in the unknown region (points less than m), we guess correctly.
  # Otherwise, we get it right with probability equal to the ratio of the bigger half of the unknown region to the whole unknown region.

  # Wow, I'm an idiot! Look at this bullshit. I can't believe I did this.
  # If you guess, you get this recursion.
  # How to handle this? Do a bunch of iterations? You just need to unroll this until the value of value_function[m_index] drops out.
  # The value function is going to be the solution to an equation.
  # However, we don't know the coefficient of the value_function[m_index]
  # because it's inside the max. 
  # If I keep iterating this, it should converge, right?
  # Why would it? if value_function starts at 0, it goes to some higher value, which forces a higher value-function again... it should increase.
  return reward + discount_factor * value_function[m_index]
  # The value of m (and thus m_index) does not change.

def query_value(m_index, x_index):
  m = discretization[m_index]
  x = discretization[x_index]
  if x >= m:
    # No sane strategy queries in this case--it learns nothing and forgoes reward 1. 
    return discount_factor * value_function[m_index]
  else:
    # With probability x, we reduce the unknown region to x, and with probability m - x, we reduce the unknown region to m - x.
    assert x_index < m_index 
    return discount_factor * ((x/m) * value_function[x_index] + ((m - x)/m) * value_function[m_index - x_index])

def average_max_value(m_index):
  value = 0
  for x_index in range(N):
    q = query_value(m_index, x_index)
    g = guess_value(m_index, x_index)
    value_of_opt_action = max(q, g)
    # if m_index == 0:
    #   print(value_of_opt_action)
    value += (1 / N) * value_of_opt_action
  return value 

# Some arithmetic isn't quite right. Todo: try on small array and look at numbers by hand. You're getting a reward of 200, which is totally impossible with a discount factor of 0.99--the max possible reward is 100. 
for m_index in range(N):
  print()
  print(m_index)
  print()
  squared_error = 1
  while squared_error > 10e-10:
    value = average_max_value(m_index)
    squared_error = (value - value_function[m_index])**2
    print(value - value_function[m_index])

  # if m_index < min_query_index:
  #   agree = abs(value - value_function[m_index]) < 10e-1
  #   if not agree:
  #     print("Index: " + str(m_index))
  #     print("Always guess value: " + str(value_function[m_index]))
  #     print("Recomputed value: " + str(value))
  #   assert agree
  
    value_function[m_index] = value 

print("Final value function: " + str(value_function))
plt.plot(value_function)
plt.show()









