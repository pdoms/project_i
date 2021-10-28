
import sys
sys.path.append("..")
from tools.linear_regression.linear_regression import LinReg

weight = [241, 162, 212, 220, 206, 152,183,167,175,156,186,213,167,189,186,172, 196]
height = [73, 68, 74, 71,69,67,68,68,67,63,71,71,64,69,69,67,72]


reg = LinReg(weight, height, ['height', 'weight'])
reg.fit()
results = reg.analyse(se_samples=10)



#predict
#make polynominal? 
#add_column
#remove_column
#evaluate --- but only when I ran some tests



