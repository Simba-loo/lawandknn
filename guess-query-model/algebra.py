from sympy import *

import numpy as np
import matplotlib.pyplot as plt 

m, alpha, beta, arrival = symbols('m alpha beta arrival', nonnegative=True)
delta_m = Symbol("\Delta m")
V = Function("V")
half = Integer(1)/2
std_namespace = {
  "m" : m,
  "alpha" : alpha,
  "beta" : beta,
  "arrival" : arrival,
  "V" : V,
  "delta_m" : delta_m,
}

def unpack(namespace):
  return namespace["m"], namespace["alpha"], namespace["beta"], namespace["arrival"], namespace["V"], namespace["delta_m"]

def m0(beta=beta):
  return 4 * (1 - beta) / beta 

def always_guess_value(namespace = std_namespace):
  m, beta = namespace["m"], namespace["beta"]
  return (1 - m / 4) / (1 - beta) 


def query_value(namespace = std_namespace):
  m, alpha, beta, arrival, V, delta_m = unpack(namespace)
  return beta * ((arrival/m) * V(arrival) + \
    ((m - arrival)/m) * V(m - arrival))

def unknown_guess_value(namespace = std_namespace):
  m, alpha, beta, arrival, V, delta_m = unpack(namespace)
  return half + (half - arrival/m) + beta * V(m)

def known_guess_value(namespace = std_namespace):
  m, alpha, beta, arrival, V, delta_m = unpack(namespace)
  return 1 + beta * V(m)

# def query_value_if_always_guess_after(namespace = std_namespace):
#   return query_value_with_known_values_after(always_guess_value, namespace)

def query_value_with_known_values_after(values_after, namespace = std_namespace):
  return query_value(namespace) \
    .subs(V(arrival), values_after(namespace).subs(m, arrival)) \
    .subs(V(m - arrival), values_after(namespace).subs(m, m - arrival))

def start_querying_point(namespace = std_namespace):
  m, alpha, beta, arrival, V, delta_m = unpack(namespace)
  return (m / 2) - (alpha * (m - m0(beta)))

# def query_expectation_if_always_guess_after(namespace = std_namespace):
#   return query_expectation_if_known_values_after(always_guess_value, namespace)

def query_expectation(namespace = std_namespace):
  m, alpha, beta, arrival, V, delta_m = unpack(namespace)

  return 2 * Integral(query_value(namespace), (arrival, start_querying_point(namespace), m/2))

def query_expectation_with_known_values_after(known_values, namespace = std_namespace):
    return 2 * Integral(query_value_with_known_values_after(known_values, namespace), (arrival, start_querying_point(namespace), m/2))

def unknown_guess_expectation(namespace = std_namespace):
  m, alpha, beta, arrival, V, delta_m = unpack(namespace)
  return 2 * Integral(half + (half - arrival/m) + beta * V(m), (arrival, 0, start_querying_point(namespace)))

def known_guess_expectation(namespace = std_namespace):
  m = namespace["m"]
  return (1 - m) * known_guess_value(namespace)

def recursive_value_function(namespace = std_namespace):
  m = namespace["m"]
  return query_expectation(namespace) + unknown_guess_expectation(namespace) + known_guess_expectation(namespace)

def recursive_value_function_with_known_values_after_query(known_values, namespace = std_namespace):
  m = namespace["m"]
  return query_expectation_with_known_values_after(known_values, namespace) + unknown_guess_expectation(namespace) + known_guess_expectation(namespace)

def value_function_solution_with_known_values_after_query(known_values, namespace = std_namespace):
  V, m = namespace["V"], namespace["m"]
  recursive_value_function_w_known_values = recursive_value_function_with_known_values_after_query(known_values, namespace)
  solution_set = solveset(recursive_value_function_w_known_values.doit() - V(m), V(m))
  solutions = list(solution_set)
  assert(len(solutions) == 1)
  return solutions[0]

def value_function_solution_if_always_guess_after_query(namespace = std_namespace):
  return value_function_solution_with_known_values_after_query(always_guess_value, namespace)

def parameterize_in_terms_of_delta_m(expr, namespace=std_namespace):
  m, alpha, beta, arrival, V, delta_m = unpack(namespace)
  return expr.subs(m, m0(beta) + delta_m)












