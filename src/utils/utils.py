import math

def counter(x):
    results = {}
    for i in x:
        if results.get(i) != None:
            results[i] += 1
        else:
            results[i] = 1
    return results


def vector_add(v,w):
    return [i + j for i,j in zip(v,w)]

def vector_sub(v,w):
    return [i - j for i,j in zip(v,w)]

def vector_sum(vs):
    result = vs[0]
    for vector in vs[1:]:
        result = vector_add(result, vector)
    return result

def scalar_mltpy(c,v):
    return [c * i for i in v]

def dot(v,w):
    return sum(i * j for i,j in zip(v,w))

def sum_of_squares(v):
    return dot(v, v)

def magnitude(v):
    return math.sqr(sum_of_squares(v))

def squared_distance(v,w):
    return sum_of_squares(vector_sub(v,w))

def distance(v,w):
    return magnitude(vector_sub(v, w))



def mean(v):
    return sum(v)/len(v)

def median(v):
    n = len(v)
    sorted_v = sorted(v)
    mid = int(n/2)
    if n % 2 == 1:
        return sorted_v[mid]
    else:
        return mean([sorted_v[mid], sorted_v[mid-1]])


def de_mean(x):
    x_bar = mean(x)
    return [i - x_bar for i in x]

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








