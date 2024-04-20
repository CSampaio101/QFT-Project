import numpy as np
from matplotlib import pyplot as plt
import scipy as sp
from scipy import special
import math
import random

def omega(k, m):
  return np.sqrt(np.power(np.linalg.norm(k), 2) + np.power(m, 2))

L, m = 10, 1

def k(n, m):
  return np.pi / L * np.array([n, m])

def Hermite(n):
  H_m = sp.special.hermite(n, monic=False)
  print(H_m)

  # x = np.linspace(-3, 3, 100)
  # y = H_m(np.sqrt(omega(k, m)) * x)
  # plt.plot(x, y)
  # plt.title("Monic Hermite polynomial of degree 3")
  # plt.xlabel("x")
  # plt.ylabel("H_3(x)")

  # print(list(zip(x, y)))

  # plt.show()

  return y

# Hermite(1)


def prob_ampl(n, N, M, plot = False):
  phi_k = np.linspace(-3, 3, 1000)
  H_m = sp.special.hermite(n, monic=False)

  y = 1 / np.power(2, n) * math.factorial(n) * np.sqrt((omega(k(N,M), m) / (np.pi))) * np.exp(-omega(k(N,M), m) * np.power(phi_k, 2)) * np.power(H_m(np.sqrt(omega(k(N,M), m)) * phi_k), 2)

  # plt.axvline(x=3 / np.sqrt(2 * omega(k, m)), color='g', linestyle='-')

  if plot: plt.plot(phi_k, y, 'r-')

  # plt.show()

  return max(y)
#Step 2 Create a box and find point lying under dist. We take the walls for the box to be at x=+-2 since I couldn't find any cases where that width didn't contain most of the distribution




#Note: We label the arbitrary mode as (N,M,L) for the sake of simplicity. The small "n" will be used to denote excitation (n=0 is the ground state)

def box(n, N, M, plot = False):
  y_min, y_max = 0, prob_ampl(n, N, M, plot)
  x_min, x_max = -2, 2
  rand_x = rand_y = float('inf')
  iteration = 0
  H_m = sp.special.hermite(n, monic=False)

  while iteration == 0 or rand_y > y:

    rand_x, rand_y = random.uniform(x_min, x_max), random.uniform(y_min, y_max)

    y = 1 / np.power(2, n) * math.factorial(n) * np.sqrt((omega(k(N, M), m) / (np.pi))) * np.exp(-omega(k(N, M), m) * np.power(rand_x, 2)) * np.power(H_m(np.sqrt(omega(k(N, M), m)) * rand_x), 2)

    # print(rand_x, rand_y, y, iteration)
    if plot: plt.plot(rand_x, rand_y, 'bo')
    iteration += 1

  if plot:
    plt.axhline(y=y_max, color='g', linestyle='-')
    plt.axvline(x=-2, color='g', linestyle='-')
    plt.axvline(x=2, color='g', linestyle='-')

    plt.xlabel('Phi_n')
    plt.ylabel('Psi^2')
    plt.title('Probability Distibution')

  return rand_x

plt.subplot(1,2,1)
box(0, 1, 1, True)
plt.subplot(1,2,2)
box(1, 1, 1, True)
plt.show()

def phi(x, y, z, MAX, NUM_POINTS, ground = True):
  phi_x = 0

  for n in range(1, MAX + 1):
    for m in range(1, MAX + 1):
        rand_x = box(0, n, m) if ground else box(random.randint(0, 1), n, m)
        phi_x += rand_x * np.power(2 / L, 2 / 2) * np.sin((n * np.pi * x) / L) * np.sin((m * np.pi * y) / L)

  if np.all(np.absolute(phi_x) < (10 ** (-15))): return np.zeros((NUM_POINTS, NUM_POINTS))

  # print(np.max(np.absolute(phi_x)))
  # print(np.min(np.absolute(phi_x)))

  # print(phi_x)
  # print(np.absolute(phi_x))

  return phi_x

NUM_POINTS = 100

# Generating data
x = np.linspace(0, L, NUM_POINTS)
y = np.linspace(0, L, NUM_POINTS)
X, Y = np.meshgrid(x, y)

def contour(z):
  Z = phi(X, Y, z, 10, NUM_POINTS, False)

  # Creating contour plot

  if z == 0 or z == L:
    contour = plt.contourf(X, Y, Z)
  else:
    contour = plt.contour(X, Y, Z)

  plt.colorbar(contour, label='Z-values')

  # Adding labels and title
  plt.xlabel('X-axis')
  plt.ylabel('Y-axis')
  plt.title('Contour Plot')

  # Displaying the plot
  # plt.show()

contour(1)
plt.show()