
import sys
sys.path.append("..")
from tools.linear_regression.linear_regression import LinReg

weight = [241, 162, 212, 220, 206, 152,183,167,175,156,186,213,167,189,186,172, 196]
height = [73, 68, 74, 71,69,67,68,68,67,63,71,71,64,69,69,67,72]
shoesize = [8,9,12,16,12,14,8,8,10,11,13,13,7, 8, 12, 16, 14]



reg = LinReg(weight, height, ['height', 'weight'])

reg.add_variable(shoesize, 'shoesize')
print(reg.x)
print(reg.fit())
results = reg.analyse(se_samples=10)



#add_column
#remove_column
#evaluate --- but only when I ran some tests



