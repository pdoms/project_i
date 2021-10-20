from .linear_regression import least_squares, predict, mean, std
import random
import math 

def rss(x,y,alpha, beta):
    data = zip(x,y)
    return sum([(yi - predict(xi, alpha, beta))**2 for xi, yi in data])

def tss(y):
    y_bar = mean(y)
    return sum([(yi - y_bar)**2 for yi in y])
    

def ess(x,y,alpha, beta):
    y_bar = mean(y)
    return sum([(predict(xi, alpha, beta) - y_bar)**2 for xi in x])

def r_square(x,y,alpha, beta):
    return 1 - rss(x,y,alpha, beta) / tss(y)

def adj_r_square(x,y,alpha, beta):
    n = len(y)
    return 1 - (rss(x,y,alpha, beta)/(n-2))/(tss(y)/(n-1))

#sqrt [ Σ(yi – ŷi)2 / (n – 2) ] / sqrt [ Σ(xi – x)2 ]
def standard_error_of_regression(x,y, alpha, beta):
    return math.sqrt(rss(x,y,alpha, beta) / (len(y) -2))

def zipp(x,y):
    zipped = []
    for n, e in enumerate(x):
        zipped.append([e, y[n]])
    return zipped

def unzipp(data):
    uz_1 = []
    uz_2 = []
    for i, ii in data:
        uz_1.append(i)
        uz_2.append(ii)
    return uz_1, uz_2





def sample_data(data):
    return random.choices(data, k=len(data))


def bootstrap(data, callback, num_samples, n):
    return [callback(sample_data(data)) for _ in range(num_samples)]

def bootstrap_least_squares(sample_data):
    sample_x, sample_y = unzipp(sample_data)
    return least_squares(sample_x, sample_y)


def standard_error(x,y, num_samples):
    betas = bootstrap(zipp(x,y), bootstrap_least_squares, num_samples, n=len(y))
    return [std([beta[i] for beta in betas]) for i in range(2)]
  



