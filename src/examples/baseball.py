#Major league baseball player statistics, minimum 400 at-bats		
#Original source:  Lahman Baseball Database 		
#http://baseball1.com/statistics/		
#Source of this file:		
#http://regressit.com/data		



import pandas as pd
import os
import sys
import random


sys.path.append("..")
from tools.analysis import anova_lin, correlation_matrix, describe, variable_summary_lin
from tools.regression import lin_least_squares_fit, l_reg
from tools.utils import normalize, scale_down
from tools.methods import error_l_reg, l_reg_squared_error, l_reg_squared_error_gradient, predict_l_reg
import matplotlib.pyplot as plt






dir_path = os.path.dirname(os.path.realpath(__file__))
source = os.path.join(dir_path, 'data', 'baseball_small.csv')

df = pd.read_csv(source)
data = df[['BattingAverage', 'BattingAverageLAG1', 'CumulativeAverageLAG1']]
###### from scratch ######

###convert to python list
bat_avg = data['BattingAverage'].to_list()
bat_avg_l1 = data['BattingAverageLAG1'].to_list()
cum_avg_l1 = data['CumulativeAverageLAG1'].to_list()


###metrics
d = {'BatAvg': bat_avg, 'BatAvgL1': bat_avg_l1, 'CumAvgL1': cum_avg_l1}
#describe(d)
#correlation_matrix(d)
scaler = scale_down(bat_avg)
bat_avg_scaled = scaler.scaled_down
cum_avg_l1_scaled = scaler.scale_down(cum_avg_l1)
alpha, beta = l_reg(cum_avg_l1_scaled, bat_avg_scaled)

variable_summary_lin(alpha, beta, bat_avg_scaled, cum_avg_l1_scaled)

#anova_lin(cum_avg_l1_scaled, bat_avg_scaled, alpha, beta)



###visualization
#fig, ax = plt.subplots(1,2, figsize=(10,7))
#ax[0].scatter(bat_avg, bat_avg_l1)
#ax[1].scatter(bat_avg, cum_avg_l1)
#print(plt.show())






###### with packages #####





