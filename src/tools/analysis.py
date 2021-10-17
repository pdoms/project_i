from .utils import correlation, de_mean, mean, std, normal_cdf, median, variance
from .methods import error
import random


def describe(data):
    headings = ['column', 'mean', 'std', 'median', 'min', 'max', 'variance', 'length']
    space = ''
    print(f'{headings[0]:{10}} {headings[1]:{13}} {headings[2]:{13}} {headings[3]:{13}} {headings[4]:{13}} {headings[5]:{13}} {headings[6]:{13}} {headings[7]:{13}}')
    print(f'{space:.>{90}}')
    for key, value in data.items():
        print('')
        print(f'{key:{10}} {mean(value):2.4f} {space:{4}} {std(value):2.4f} {space:{5}} {median(value):2.4f} {space:{4}} {min(value):2.4f} {space:{4}} {max(value):2.4f} {space:{4}} {variance(value):2.4f} {space:{4}} {len(value)}')

def correlation_matrix(data):
    processed = []
    for k_1, v_1 in data.items():
        for k_2, v_2 in data.items():
            if k_1+k_2 not in processed:
                print(f'{k_1} vs {k_2} ---> {correlation(v_1, v_2)}')
                processed.append(k_2+k_1)




def tss(y):
    return sum(v**2 for v in de_mean(y))


def r_squared(x,y,beta):
    ssr = sum(error(xi, yi, beta) ** 2
        for xi, yi in zip(x,y))
    return 1.0 - ssr / tss(y)

def bootstrap_sample(data):
    return [random.choice(data) for _ in data]


def bootstrap_statistic(data, stats_fn, num_samples):
    return [stats_fn(bootstrap_sample(data)) for _ in range(num_samples)]
#needed 
# (SQRT(1 minus adjusted-R-squared)) x STDEV.S(Y). // Standard error of regression = STDEV.S(errors) x SQRT((n-1)/(n-2))
# adjusted r2

def p_value(beta_hat_j, sigma_hat_j):
    if beta_hat_j > 0:
        return 2 * (1 - normal_cdf(beta_hat_j / sigma_hat_j))
    else:
        return 2 * normal_cdf(beta_hat_j / sigma_hat_j)


