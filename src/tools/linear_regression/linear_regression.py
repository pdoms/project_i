import time
import math
import random
from columnar import columnar
from scipy.stats import t
from datetime import datetime






#helper functions 
def scale_for_print(arr, scale):
    return [i / scale for i in arr]

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

def vector_sub(v,w):
    return [i - j for i,j in zip(v,w)]

def scalar_mltply(c,v):
    return [c * i for i in v]

def drop(v, idxs):
    if len(idxs) < len(v[0]):
        for item in v:
            for i in idxs:
                del item[i]
        return v
    else:
        raise IndexError('Out of Range')



def shuffle(data):
    indexes = [i for i, _ in enumerate(data)]
    random.shuffle(indexes)
    for i in indexes:
        yield data[i]


def least_squares(x,y):
    beta = correlation(x,y) * std(y) / std(x)
    alpha = mean(y) - beta * mean(x)
    return alpha, beta

def predict(xi,alpha, beta):
    return beta * xi + alpha

class scale_down():

    def __init__(self, X):
        self.X = X
        self.scale = self.scale_initial()
        self.scaled_down = self.scale_down(X)
    
    def scale_initial(self):
        maximum = max(self.X)
        factor = 10
        if maximum > 1:
            while maximum/factor > 1:
                factor *= 10
        self.scale = factor
        return self.scale
            
def rescale_constant(beta, scale):
    const = beta[0]*scale
    beta_ = beta[1:]
    beta_.insert(0, const)
    return beta_

def get_scale(v):
    maximum = max(v)
    if maximum <= 1:
        return 1
    factor = 10
    while maximum/factor > 1:
        factor *= 10
    return factor

def scale_down(x,y,scale):
    ys = [i/scale for i in y]
    xs = []
    for item in x:
        upd = [1] + [i/scale for i in item[1:]]
        xs.append(upd)
    return xs, ys

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


def bootstrap(data, callback, num_samples):
    return [callback(sample_data(data)) for _ in range(num_samples)]



def minimize_stochastic(method, method_gradient, x,y, theta_0, alpha_0=0.01, max_iter=80):
    scale = get_scale(y)
    x_scaled, y_scaled = scale_down(x,y, scale)
    iterations = 0
    data = list(zip(x_scaled,y_scaled))
    theta = theta_0
    alpha = alpha_0
    min_theta, min_err_val = None, float('inf')
    while iterations <= max_iter:
        err_val = sum(method(xi, yi, theta) for xi, yi in data)
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
    return rescale_constant(min_theta, scale)



def write_y_stats(metrics, name='...'):
    written = 'Description of ' + name + '\n' + (34 * '-') + '\n'
    for key, value in metrics.items():
        written += f'{key:{20}} {value:2.6f}\n'
    written += 34 * '-' + '\n'
    return written

def key_to_print(key):
    upd_key = ''
    next_upper = False
    for i, char in enumerate(key):
        if next_upper:
            upd_key+= char.upper()
            next_upper = False
            continue
        if i == 0:
            upd_key += char.upper()
        elif char == '_':
            upd_key += ' '
            next_upper = True
            
        else: 
            upd_key += char
    return upd_key




test_data = ({'regresion': {'df': 1, 'sum_squares': None, 'mean_squares': 91.83029187838947}, 'residual': {'df': 14, 'sum_squares': 46.46057290649245, 'mean_squares': 3.3186123504637464}, 'total': {'df': 16, 'sum_squares': 137.76470588235293, 'mean_squares': 8.610294117647058}, 'regression_f': {'regression_f': 27.67129214882751}},  {'r_square': 0.6627541676300719, 'adj_r_square': 0.998594809031792, 'multiple_r': 0.8140971487667009, 'std_error_regression': 0.856139519501639}, {'head': ['height', 'Coefficient', 'SE', 'T_Stat', 'CV', 'P_Val', 'Lower 95.0%', 'Upper 95.0%'], 'coefficients': [50.71000204631936, 0.09705359506584726], 'se': [3.6240239802629026, 0.01908020536374769]})

def write_analysis(anova, reg_stats, reg_analysis, note=''):
    now = datetime.now()
    overview = now.strftime("%d/%m/%Y %H:%M:%S") + '\n'
    delim = ('- '*39) + '\n'
    table_reg_rows = ['REGRESSION STATS', ' ']
    d_1 = []
    for k,v in reg_stats.items():
        d_1.append([key_to_print(k), v])
    t_1 = columnar(d_1, table_reg_rows)
    overview += t_1
    overview += delim
    regression = anova.get('regression').get('tup')
    residual = anova.get('residual').get('tup')
    total = anova.get('total').get('tup')
    reg = ['ANOVA', 'DF', 'SS', 'MS']
    d_2 = [
        ['Regression', regression[0], regression[1], regression[2]],
        ['Residual', residual[0], residual[1], residual[2]],
        ['Total', total[0], total[1], total[2]],
    ]
    t_2 = columnar(d_2, reg)
    overview += t_2
    overview += delim
    overview += '  COEFFICIENT ANALYSIS\n'
    d_3_all = reg_analysis.get('values')
    t_3 = columnar(d_3_all[1:], d_3_all[0])
    overview += t_3
    overview += 'Critical Value: ' + str(reg_analysis.get('cv')) + '\n\n'
    if len(note) > 0:
        overview += 'NOTES: ' + note + '\n'
    else:
        overview += 'NOTES: -- \n'
    overview += ('-'*30) + ' END OF ANALYSIS ' + ('-'*30)
    return overview

    
def get_column(A,j):
    return [Ai[j] for Ai in A]

def shape(A):
  num_rows = len(A)
  num_cols = len(A[0] if A else 0)
  return num_rows, num_cols

def transpose(A):
  return [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]

def zero_matrix(A,B):
  rows, _ = shape(A)
  _, cols = shape(B)
  return [[0.0 for j in range(cols)] for i in range(rows)]

def vector_to_matrix(V):
  if type(V[0]) != list:
    return [[i] for i in V]
  else:
    return V 


def matrix_mltply(A,B):
  A = vector_to_matrix(A)
  B = vector_to_matrix(B)
  result = zero_matrix(A,B)
  for i in range(len(A)):
    for j in range(len(B[0])):
      for k in range(len(B)):
        result[i][j] += A[i][k] * B[k][j]
  return result

def shapeMatrix(rows, cols, fn):
  return [[fn(i, j) for j in range(cols)] for i in range(rows)]

def diagonal(i,j):
  return 1.0 if i==j else 0.0

def matrix_inv(A):
  n = len(A)
  r,c = shape(A)
  I = shapeMatrix(r,c, diagonal)
  indices = list(range(n))
  for fd in range(n):
    fd_scaler = (1/A[fd][fd]) if A[fd][fd] != 0 else 0 
    for j in range(n):
      A[fd][j] *= fd_scaler 
      I[fd][j] *= fd_scaler 
    for i in indices[0:fd] + indices[fd+1:]:
      crScaler = A[i][fd]
      for j in range(n):
        A[i][j] = A[i][j] - crScaler * A[fd][j]
        I[i][j] = I[i][j] - crScaler * I[fd][j]
  return I

def flatten(V): 
    return [i[0] for i in V]

def multiple_least_squares(x,y):
  x_transp = transpose(x)
  gram_inv = matrix_inv(matrix_mltply(x_transp, x))
  moment_matrix = matrix_mltply(x_transp, y)
  return flatten(matrix_mltply(gram_inv, moment_matrix))





    

def p_value(beta, se, df):
    return float((1 - t.cdf(abs(beta/se), df)) * 2)
    
    
def upper_bound(beta, se, cv):
    return (beta + (se * cv))
def lower_bound(beta, se, cv):
    return (beta - (se * cv))
    

    
    
    

class LinReg:
    def __init__(self, x, y, columns=[]):
        self.y_raw = y
        self.x_raw = x
        self.is_simple = type(x[0]) != list
        self.x, self.y = self.__pre_check(x,y)
        self.columns = columns
        self.beta = []
        self.n = len(self.y)
        self.k = None
        self.df = None
        self.rss = None
        self.ess = None
        self.tss = None
        self.r_sq = None
        self.adj_r_sq = None
        self.ser = None
        self.se = None
        self.reg_f = None
        self.data_fit_option = 'ordinary'
        self.p_vals = None
        self.t_stats = None
        self.cv = None
        self.lowers = None
        self.uppers = None

    

    def __pre_check(self, x,y):
        if type(x[0]) != list:
            x = [[1.0,i] for i in x]
        else: 
            for i in range(len(x)):
                x[i].insert(0, 1.0)
        return x,y   
            
    def __pred(self, xi, _beta):
        return  dot(xi, _beta)
        
    def __error(self,xi, yi, _beta):
        return yi - self.__pred(xi,_beta)
    
    def __squared_error(self, xi, yi, _beta):
        return self.__error(xi, yi, _beta)**2
    
    def __squared_error_gradient(self, xi, yi, _beta):
        return [-2 * xij * self.__error(xi, yi, _beta)
                for xij in xi]
    
    def __estimate_beta(self, alpha_0, max_iter):
        _beta = [random.random() for xi in self.x[0]]
        
        return minimize_stochastic(self.__squared_error, 
                                    self.__squared_error_gradient,
                                    self.x, 
                                    self.y,
                                    _beta,
                                    alpha_0,
                                    max_iter
                                    )

    def __ordinary_least_squares(self):
        if self.is_simple:
            alpha, beta = least_squares(self.x_raw, self.y_raw)
            return [alpha, beta]
        else: 
            return multiple_least_squares(self.x, self.y)
        


    def fit(self, alpha_0=0.0001, max_iter=80, method='ordinary'):
        self.data_fit_option = method
        if method == 'ordinary':
            self.beta = self.__ordinary_least_squares()
        else:
            self.beta = self.__estimate_beta(alpha_0, max_iter)
        self.k = len(self.beta)
        self.df = self.n - self.k
        return self.beta

    def y_stats(self, report=True, write=''):
        metrics =  {
            'length': len(self.y_raw),
            'mean': mean(self.y_raw),
            'median': median(self.y_raw),
            'standard_deviation': std(self.y_raw),
            'variance': variance(self.y_raw),
        }
        written = write_y_stats(metrics, self.columns[0])
        if report:
            print(written)
        if len(write) > 0:
            with open(write, 'a') as f:
                f.write(written)
        return metrics

    def analyse(self, ci=0.95, se_samples=100, report=True, write='', add_note=False):
        if self.data_fit_option != 'ordinary':
            print(f'Analyizing regression... standard error collects {se_samples} data-samples... so this might take a while... sorry.')
        anova = {
            'regression': {
                'df': self.k -1,
                'sum_squares': self.__estimated_sum_squares(),
                'mean_squares': self.ess / (self.k -1),
                'tup': (self.k-1, self.ess, self.ess/(self.k))
            },
            'residual': {
                'df': self.df,
                'sum_squares': self.__residual_sum_squares(),
                'mean_squares': self.rss / (self.df - 1),
                'tup': (self.df, self.rss, self.rss/(self.df))
            },
            'total': {
                'df': self.n -1,
                'sum_squares': self.__total_sum_squares(),
                'mean_squares': self.tss / (self.n -1),
                'tup': (self.n -1, self.tss, self.tss/(self.n-1))
            },
            'regression_f': {
                'regression_f': self.__regression_f()
            } 
        }
        reg_stats = {
            'r_square': self.__r_squared(),
            'adj_r_square': self.__adj_r_squared(),
            'multiple_r': math.sqrt(self.r_sq),
            'std_error_regression': self.__standard_error_of_regression()
        }
        self.__standard_error(se_samples)
        cv, vals = self.__create_coefficient_analysis(ci)
        reg_analysis = {'values': vals, 'cv': cv}
        note = ''
        if add_note:
            if len(write) == 0:
                print('In order to add a note, a filename must be provided via the "write" argument.')
            else: 
                note = input('Add a note or leave blank if you want to skip: ')
        written = write_analysis(anova, reg_stats, reg_analysis, note)
        if report:
            print('\n\n', written)
        if len(write) > 0:
            with open(write, 'a') as f:
                f.write(written)
        return (anova, reg_stats, reg_analysis, {'note': note})
    
    def predict(self, x):
        if type(x) != list:
            return self.beta[0] + self.beta[1]*x
        else:
            return dot(self.beta, x)

    def __create_coefficient_analysis(self, ci=0.95):
        labels = self.columns.copy()
        labels.insert(1, '_const')
        beta_ = self.beta.copy()
        beta_.insert(0, 'Coefficients')
        se_ = self.se.copy()
        se_.insert(0, 'SE')
        beta_and_se = zipp(self.beta, self.se)
        self.t_stats = [(b/s) if s != 0 else 0 for b,s in beta_and_se]
        self.t_stats.insert(0, 'T-Stat')
        sl = (1 - 0.95) / 2
        ci_pct = ci*100
        self.cv = t.ppf(1-sl, self.df-1)
        self.p_vals = [p_value(b,s, self.df-1) for b,s in beta_and_se]
        self.p_vals.insert(0, 'P-Value')
        
        self.lowers = [ lower_bound(b, s, self.cv) for b,s in beta_and_se]
        self.uppers = [upper_bound(b,s,self.cv) for b,s in beta_and_se]
        self.lowers.insert(0, f'Lower {ci_pct:2.1f}%')
        self.uppers.insert(0, f'Upper {ci_pct:2.1f}%')
        data_ = []
        for i, label in enumerate(labels):
            data_.append([label, beta_[i], se_[i], self.t_stats[i], self.p_vals[i], self.lowers[i], self.uppers[i]])
        return (self.cv, data_)
        






    def __r_squared(self):
        self.r_sq = 1 - self.rss / self.tss
        return self.r_sq

    def __adj_r_squared(self):
        self.adj_r_sq = 1 - ((self.rss/(self.n - 2))/(self.tss/(self.n-1)))
        return self.adj_r_sq

    def __standard_error_of_regression(self):
        self.ser = math.sqrt(self.rss/(self.n - 2))
        return self.ser

    #ANOVA 


    def __residual_sum_squares(self):
        data = zip(self.x, self.y)
        self.rss = sum([(yi - self.__pred(xi, self.beta))**2 for xi, yi in data])
        return self.rss


    def __estimated_sum_squares(self):
        y_bar = mean(self.y)
        self.ess = sum([(self.__pred(xi, self.beta) - y_bar)**2 for xi in self.x])
        return self.ess

    
    def __total_sum_squares(self):
        y_bar = mean(self.y)
        self.tss = sum([(yi - y_bar)**2 for yi in self.y])
        return self.tss
    
    def __regression_f(self):
        self.reg_f = (self.ess/(self.k -1))/(self.rss/(self.df -1))
        return self.reg_f

    #coefficients


    def __estimate(self, x,y, alpha_0, max_iter):
        _beta = [random.random() for xi in x[0]]
        return minimize_stochastic(self.__squared_error, 
                                    self.__squared_error_gradient,
                                    x, 
                                    y,
                                    _beta,
                                    alpha_0,
                                    max_iter
                                    )
    
    def __bootstrap_beta(self, sample_data):
        sample_x, sample_y = unzipp(sample_data)
        return self.__estimate(sample_x, sample_y, 0.01, 80)

    def __estimate_ols(self, x,y):
        if self.is_simple:
            flat_x = [i[1] for i in x]
            alpha, beta = least_squares(flat_x, y)
            return [alpha, beta]
        else: 
            return multiple_least_squares(x, y)

    def __bootstrap_beta_ols(self, sample_data):
        sample_x, sample_y = unzipp(sample_data)
        return self.__estimate_ols(sample_x, sample_y)

    
    
    def __standard_error(self, num_samples):
        if self.data_fit_option == 'ordinary':
            beta_estimates = bootstrap(zipp(self.x, self.y), self.__bootstrap_beta_ols, num_samples)
            self.se = [std([b[i] for b in beta_estimates]) for i in range(self.k)]
        else:
            beta_estimates = bootstrap(zipp(self.x, self.y), self.__bootstrap_beta, num_samples)
            self.se = [std([b[i] for b in beta_estimates]) for i in range(self.k)]
        return self.se
    
    def add_variable(self, variables, id=''):
        if len(self.x[0]) >= 2:
            self.is_simple = False
        self.columns.append(id)
        for i in range(self.n):
            self.x[i].append(variables[i])

        return self.x
    
    def make_dummy(self, src, label_base='', initial=''):
        cats = len(set(src))
        dummy = [[0]*cats for _ in range(len(src))]
        for no,i in enumerate(src):
            dummy[no][i] = 1        
            for item in dummy[no]:
                self.x[no].append(item)
        for i in range(1, cats +1):
            self.columns.append(f'd_{label_base}_cat_{i}')
        if initial != '':
            self.drop_var([initial])
        
        

  

    def drop_var(self, ids):
        idxs = [self.columns.index(id) for id in ids]
        for i in ids:
            idx = self.columns.index(i)
            del self.columns[idx]
        self.x = drop(self.x, idxs)
        return self.x


    def evaluate(self):
        print('I am checking R-Square adjusted...\n')
        time.sleep(.6)
        if self.adj_r_sq == None:
            print('run analysis first!')
            return
        if self.adj_r_sq <= 0.5:
            print('R-Square (adjusted) seriously low, no string explanatory power\n')
        elif self.adj_r_sq > 0.5 <= 0.80:
            print('R-Square (adjusted) quite good, could be better.\n')
        else: 
            print('R-squared (adjusted) sufficiently high, guess that works.\n')
        time.sleep(.6)
        print('Interpreting coefficient... mind that I can only report the obvious and that I\'m weak if units are too diverse in quantity, you know what I mean.\n')
        time.sleep(.6)
        for i, b in enumerate(self.beta):
            if i == 0:
                print('I ignore the y-intercept, hope that\'s ok. Anyways...\n')
            else:
                dir = 'increase'
                if b < 0:
                    dir = 'decrease'
                print(f'If all else equal {self.columns[0]} will {dir} by {b} units if {self.columns[i]} changes by 1 unit.\n')
                time.sleep(.6)
        print('T-test and Confidence Intervals are next, and last. I\'m getting tired. Next time do it yourself, ok?')
        for i, b in enumerate(self.beta):
            if i == 0:
                print('Still don\'t care much about the constant term.\n')
            else: 
                if self.p_vals[i+1] < 0.05:
                    print(f'Looks good, {self.columns[i]}\'s P-Value is smaller than 0.05\n')
                else:
                    print(f'Variable {self.columns[i]} is bigger than 0.05, I wouldn\'t trust it. Really, null hypothesis is should not be rejected for this one.\n')
                time.sleep(.3)
                if abs(self.t_stats[i+1]) > self.cv:
                    print(f'Variable {self.columns[i]} passed the T-Test.\n')
                else:
                    print(f'Variable {self.columns[i]} is smaller than the critical value. Not good.\n')
                if self.lowers[i+1] < 0 and self.uppers[i+1] < 0:
                    print(f'No zero or null in confidence interval for {self.columns[i]}\n')
                elif self.lowers[i+1] > 0 and self.uppers[i+1] > 0:
                    print(f'No zero or null in confidence interval for {self.columns[i]}\n')
                else: 
                    print(f'Confidence intervall for {self.columns[i]} includes a zeror or null. Wouldn\'t rely on that one.\n')
            time.sleep(.6)
        print('I am done.\n')

                
        

    





















