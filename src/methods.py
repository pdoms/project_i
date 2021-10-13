from utils.utils import dot

def predict_mReg(xi, beta):
    return dot(xi, beta)

def error(xi, yi, beta):
    return yi - predict_mReg(xi, beta)

def squared_error(xi, yi, beta):
    return error(xi, yi, beta) ** 2

def squared_error_gradient(xi,yi,beta):
    return [-2 * xii * error(xi,yi,beta) for xii in xi]

