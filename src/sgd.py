from utils.utils import *


def stochastic_gradient_descent(method, method_gradient, x,y, theta_0, threshold_0=0.01, max_iter=100):
    iterations = 0
    data = list(zip(x,y)    )
    theta = theta_0
    threshold = threshold_0
    min_theta, min_err_val = None, float('inf')
    while iterations <= max_iter:
        err_val = sum(method(xi,yi, theta) for xi, yi in data)
        if err_val < min_err_val:
            #err val smaller than previous --> improvement
            min_theta, min_err_val = theta, err_val
            iterations = 0
            threshold = threshold_0
        else:
            #no improvement --> adjust or decrease threshold/alpha
            iterations += 1
            threshold *= 0.9
        for xi, yi in shuffle(data):
            gradienti = method_gradient(xi, yi, theta)
            theta = vector_sub(theta, scalar_mltply(threshold, gradienti))
    return min_theta



