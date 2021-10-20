import math

#helper functions 

def mean(v):
    return sum(v)/len(v)

def variance(x):
    devs = de_mean(x)
    return sum_of_squares(devs) / (len(x) - 1)

def std(x):
    return math.sqrt(variance(x))

def covariance(x,y):
    return dot(de_mean(x), de_mean(y)) / (len(x) -1)

def correlation(x,y):
    std_x = std(x)
    std_y = std(y)
    if std_x > 0 and std_y >0:
        return covariance(x,y) /std_x/std_y
    else: 
        return 0

def de_mean(x):
    x_bar = mean(x)
    return [i - x_bar for i in x]

def dot(v,w):
    return sum(i * j for i,j in zip(v,w))

def sum_of_squares(v):
    return dot(v, v)


#simple linear regression

def least_squares(x,y):
    beta = correlation(x,y) * std(y) / std(x)
    alpha = mean(y) - beta * mean(x)
    return alpha, beta

def predict(xi,alpha, beta):
    return beta * xi + alpha