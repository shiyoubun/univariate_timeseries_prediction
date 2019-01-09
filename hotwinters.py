from __future__ import division
from sys import exit
from math import sqrt
from numpy import array
from scipy.optimize import fmin_l_bfgs_b

def RMSE(params, *args):

	Y = args[0]
	type = args[1]
	rmse = 0

	if type == 'linear':

		alpha, beta = params
		a = [Y[0]]
		b = [Y[1] - Y[0]]
		y = [a[0] + b[0]]

		for i in range(len(Y)):

			a.append(alpha * Y[i] + (1 - alpha) * (a[i] + b[i]))
			b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
			y.append(a[i + 1] + b[i + 1])

	else:

		alpha, beta, gamma = params
		m = args[2]
		a = [sum(Y[0:m]) / float(m)]
		b = [(sum(Y[m:2 * m]) - sum(Y[0:m])) / m ** 2]

		if type == 'additive':

			s = [Y[i] - a[0] for i in range(m)]
			y = [a[0] + b[0] + s[0]]

			for i in range(len(Y)):

				a.append(alpha * (Y[i] - s[i]) + (1 - alpha) * (a[i] + b[i]))
				b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
				s.append(gamma * (Y[i] - a[i] - b[i]) + (1 - gamma) * s[i])
				y.append(a[i + 1] + b[i + 1] + s[i + 1])

		elif type == 'multiplicative':

			s = [Y[i] / a[0] for i in range(m)]
			y = [(a[0] + b[0]) * s[0]]

			for i in range(len(Y)):

				a.append(alpha * (Y[i] / s[i]) + (1 - alpha) * (a[i] + b[i]))
				b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
				s.append(gamma * (Y[i] / (a[i] + b[i])) + (1 - gamma) * s[i])
				y.append((a[i + 1] + b[i + 1]) * s[i + 1])

		else:

			exit('Type must be either linear, additive or multiplicative')
		
	rmse = sqrt(sum([(m - n) ** 2 for m, n in zip(Y, y[:-1])]) / len(Y))

	return y, Y, rmse

def linear(x, fc, alpha = None, beta = None):

	Y = x[:]

	if (alpha == None or beta == None):

		initial_values = array([0.3, 0.1])
		boundaries = [(0, 1), (0, 1)]
		type = 'linear'

		parameters = fmin_l_bfgs_b(RMSE, x0 = initial_values, args = (Y, type), bounds = boundaries, approx_grad = True)
		alpha, beta = parameters[0]

	s = [Y[0]]
	t = [Y[1] - Y[0]]
	y = [s[0] + t[0]]
	rmse = 0

	for i in range(len(Y) + fc):

		if i == len(Y):
			Y.append(s[-1] + t[-1])

		s.append(alpha * Y[i] + (1 - alpha) * (s[i] + t[i]))
		t.append(beta * (s[i + 1] - s[i]) + (1 - beta) * t[i])
		y.append(s[i + 1] + t[i + 1])

	rmse = sqrt(sum([(m - n) ** 2 for m, n in zip(Y[:-fc], y[:-fc - 1])]) / len(Y[:-fc]))

	return Y[-fc:], alpha, beta, rmse

def additive(x, k, fc, alpha = None, beta = None, gamma = None):

	Y = x[:]

	if (alpha == None or beta == None or gamma == None):

		initial_values = array([0.3, 0.1, 0.1])
		boundaries = [(0, 1), (0, 1), (0, 1)]
		type = 'additive'

		parameters = fmin_l_bfgs_b(RMSE, x0 = initial_values, args = (Y, type, k), bounds = boundaries, approx_grad = True)
		alpha, beta, gamma = parameters[0]

	s = [sum(Y[0:k]) / float(m)]
	t = [(sum(Y[k:2 * k]) - sum(Y[0:k])) / k]
	p = [Y[i] - s[0] for i in range(k)]
	y = [s[0] + t[0] + p[0]]
	rmse = 0

	for i in range(len(Y) + fc):

		if i == len(Y):
			Y.append(s[-1] + t[-1] + p[-k])

		s.append(alpha * (Y[i] - p[i]) + (1 - alpha) * (s[i] + t[i]))
		t.append(beta * (s[i + 1] - s[i]) + (1 - beta) * t[i])
		p.append(gamma * (Y[i] - s[i]) + (1 - gamma) * p[i])
		y.append(s[i + 1] + t[i + 1] + p[i + 1])

	rmse = sqrt(sum([(m - n) ** 2 for m, n in zip(Y[:-fc], y[:-fc - 1])]) / len(Y[:-fc]))

	return Y[-fc:], alpha, beta, gamma, rmse

def multiplicative(x, k, fc, alpha = None, beta = None, gamma = None):

	Y = x[:]

	if (alpha == None or beta == None or gamma == None):

		initial_values = array([0.0, 1.0, 0.0])
		boundaries = [(0, 1), (0, 1), (0, 1)]
		type = 'multiplicative'

		parameters = fmin_l_bfgs_b(RMSE, x0 = initial_values, args = (Y, type, k), bounds = boundaries, approx_grad = True)
		alpha, beta, gamma = parameters[0]

	s = [sum(Y[0:k]) / float(k)]
	t = [(sum(Y[k:2 * k]) - sum(Y[0:k])) / k]
	p = [Y[i] / s[0] for i in range(k)]
	y = [(s[0] + t[0]) * p[0]]
	rmse = 0

	for i in range(len(Y) + fc):

		if i == len(Y):
			Y.append((s[-1] + t[-1]) * p[-k])

		s.append(alpha * (Y[i] / p[i]) + (1 - alpha) * (s[i] + t[i]))
		t.append(beta * (s[i + 1] - s[i]) + (1 - beta) * t[i])
		p.append(gamma * (Y[i] / (s[i]) + (1 - gamma) * p[i])
		y.append((s[i + 1] + t[i + 1]) * p[i + 1])

	rmse = sqrt(sum([(m - n) ** 2 for m, n in zip(Y[:-fc], y[:-fc - 1])]) / len(Y[:-fc]))

	return Y[-fc:], alpha, beta, gamma, rmse
