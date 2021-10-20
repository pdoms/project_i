
import sys
sys.path.append("..")
from tools.linear_regression.linear_regression import least_squares
from tools.linear_regression.analysis import r_square, adj_r_square, tss, rss, ess, standard_error_of_regression
import math
weight = [241, 162, 212, 220, 206, 152,183,167,175,156,186,213,167,189,186,172, 196]
height = [73, 68, 74, 71,69,67,68,68,67,63,71,71,64,69,69,67,72]


alpha, beta = least_squares(weight, height)
print('COEFFICIENTS', alpha, beta)
""" print(tss(height))
print(rss(weight, height, alpha, beta))
print(ess(weight, height, alpha, beta)) """
print('R2', r_square(weight, height, alpha, beta))
print('multipleR', math.sqrt(r_square(weight, height, alpha, beta)))
print('adjR2', adj_r_square(weight, height, alpha, beta))
print('SE', standard_error_of_regression(weight, height, alpha, beta))




#std error

