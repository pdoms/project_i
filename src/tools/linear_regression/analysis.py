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
  
def t_test(cv, t_stat):
    return abs(cv) < t_stat








#source for beta functions wikipedia + https://malishoaib.wordpress.com/2014/04/15/the-beautiful-beta-functions-in-raw-python/
def beta(x,y):
    return math.gamma(x)*math.gamma(y)/math.gamma(x+y)

def incompletebeta(a,b,x):
    print(a, b, x)
    if x==0:
        return x
    elif x==1:
        return x
    else:
        lbeta = math.lgamma(a+b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1-x)
        if x < (a+1)/(a+a+2):
            return math.exp(lbeta) * contfractbeta(a,b,x) / a
        else:
            return 1- math.exp(lbeta) * contfractbeta(b,a, 1-x) / b

def contfractbeta(a,b,x, ITMAX=200):
    EPS = 3.0e-7
    bm = az = am = 1.0
    qab = a+b
    qap = a+ 1.0
    qam = a-1.0
    bz = 1.0-qab*x/qap

    for i in range(ITMAX+1):
        em = float(i+1)
        tem = em + em
        d = em*(b-em)*x/((qam+tem)*(a+tem))
        ap = az + d*am
        bp = bz+d*bm
        d = -(a+em)*(qab+em)*x/((qap+tem)*(a+tem))
        app = ap+d*az
        bpp = bp+d*bz
        aold = az
        am = ap/bpp
        bm = bp/bpp
        az = app/bpp
        bz = 1.0
        if abs(az-aold) < EPS*abs(az):
            return az
    print('a or b too large or given ITMAX too small for computing incomplete beta function.')

def I(a,b, xoft):
    return (incompletebeta(a,b,xoft)/(beta(a,b)))

#where df = degrees of freedom
#where xoft = df/t^2+df

def t_distribution_cdf(t, df):
    xoft = df/((t**2)+df)
    return 1 - (I((df/2), (1/2), xoft))


'''                               gamma((df+1)/2)
    t.pdf(x, df) = ---------------------------------------------------
                   sqrt(pi*df) * gamma(df/2) * (1+x**2/df)**((df+1)/2)'''

def t_pdf(x, df):
    return math.gamma((df+1)/2)/(math.sqrt(math.pi*df) * math.gamma(df/2) * (1+(x**2)/df)**((df+1)/2))