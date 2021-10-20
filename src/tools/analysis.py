from tools.regression import est_l_reg_beta
from .utils import correlation, de_mean, mean, std, normal_cdf, median, variance
from .methods import error, l_reg_sum_of_squared_errors, l_reg_tss
import random
import math

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

def sum_of_squared_error(x):
    x_bar = mean(x)
    return sum([(xi - x_bar)**2 for xi in x])

def r_squared(alpha, beta, x, y):
    return 1.0 - (l_reg_sum_of_squared_errors(alpha, beta, x,y) /
                l_reg_tss(y))
def df(n, beta):
    if isinstance(beta, list):
        return n - len(beta)
    else:
        return n - 2

def adj_r_squared(alpha, beta, x,y):
    r2 = r_squared(alpha, beta, x,y)
    return 1 - (len(x)-1)/(df(len(x), beta))*(1 - r2)

def standard_error_of_regression(alpha, beta, x,y):
    return math.sqrt(1 - adj_r_squared(alpha, beta, x,y)) * std(y)

def standard_error_mean(x):
    return std(x)/math.sqrt(len(x))

def standard_error_of_regression_alt(y,x):
    return math.sqrt(std(y)**2 + standard_error_mean(x)**2)


def mean_squared(df, sum_of_squ):
    return sum_of_squ/df

def bootstrap_beta_lin_reg(x,y):
    return bootstrap_statistic(zip(x,y), est_l_reg_beta, 100)





def anova_lin(x, y, alpha, beta):
    results = {}
    results['r2'] = r_squared(alpha, beta, x, y)
    results['adj_r2'] = adj_r_squared(alpha, beta, x,y)
    results['ss_total'] = sum_of_squared_error(y)
    results['ss_model'] = results['r2'] * results['ss_total']
    results['ss_residual'] = results['ss_total'] - results['ss_model']
    results['total_df'] = len(y) - 1
    results['model_df'] = df(len(y), beta)
    results['residual_df'] = results['total_df'] - results['model_df']
    results['ms_total'] = mean_squared(results['total_df'], results['ss_total'])
    results['ms_model'] = mean_squared(results['model_df'], results['ss_model'])
    results['ms_residual'] = mean_squared(results['residual_df'], results['ss_residual'])
    results['std_error_reg'] = standard_error_of_regression(alpha, beta, x,y)
    results['f_score'] = results['ms_model']/results['ms_residual']
    
    for k, v in results.items():
        print(k, ': ', v, '\n')

    print(standard_error_of_regression_alt(x,y))


def variable_summary_lin(alpha, beta, x,y):
    print('INTERCEPT: ', alpha)
    print('VAR: ', beta)
    print('INT STDERR (COEFF): ', bootstrap_beta_lin_reg(x,y))
    print('VAR STDERR (COEFF): ', bootstrap_beta_lin_reg(x,y))






