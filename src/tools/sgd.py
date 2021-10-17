from .utils import *


def stochastic_gradient_descent_min(method, method_gradient, x,y, theta_0, alpha_0=0.01, max_iter=80):
    iterations = 0
    data = list(zip(x,y))
    theta = theta_0
    alpha = alpha_0
    min_theta, min_err_val = None, float('inf')
    while iterations <= max_iter:
        print(iterations)
        err_val = sum(method(xi, yi, theta) for xi, yi in data)
        print(err_val, min_err_val)
        if err_val < min_err_val:
            #err val smaller than previous --> improvement
            min_theta, min_err_val = theta, err_val
            iterations = 0
            alpha = alpha_0
        else:
            #no improvement --> adjust or decrease threshold/alpha
            iterations += 1
            alpha *= 0.9
        for xi, yi in shuffle(data):
            gradienti = method_gradient(xi, yi, theta)
            theta = vector_sub(theta, scalar_mltply(alpha, gradienti))
    return min_theta



