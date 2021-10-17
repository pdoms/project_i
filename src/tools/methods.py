from .utils import dot

def predict_m_reg(xi, beta):
    return dot(xi, beta)

def error(xi, yi, beta):
    return yi - predict_m_reg(xi, beta)

def squared_error(xi, yi, beta):
    return error(xi, yi, beta) ** 2

def squared_error_gradient(xi,yi,beta):
    return [-2 * xii * error(xi,yi,beta) for xii in xi]

def predict_l_reg(alpha, beta, xi):
    return beta * xi + alpha

def error_l_reg(alpha, beta, xi, yi):
    return yi - predict_l_reg(alpha, beta, xi)

def l_reg_squared_error(xi,yi, theta):
    alpha, beta = theta
    return error_l_reg(alpha, beta, xi, yi) ** 2

def l_reg_squared_error_gradient(xi, yi, theta):
    alpha, beta = theta
    return [-2 * error_l_reg(alpha, beta, xi, yi), 
             -2 * error_l_reg(alpha, beta, xi, yi) * xi]