import pandas as pd
import matplotlib.pyplot as plt
import os
import sys


sys.path.append("..")
from tools.linear_regression.linear_regression import LinReg

#linear regression with one independent variable 

#load dataset
dir_path = os.path.dirname(os.path.realpath(__file__))
source = os.path.join(dir_path, 'data', 'weight-height.csv')
df = pd.read_csv(source)

#restrict to one gender
females = df[df['Gender'] == 'Female']

#determine independent (x) and dependent (y) variable
x = females['Weight'].tolist()
y = females['Height'].tolist()

#initialize regression class
regress = LinReg(x,y, columns=['weight', 'height'])

#describe dependent variable
regress.y_stats(report=True, write='results/weight_height.txt')

#fit data - use OLS
regress.fit(method='ordinary')
regress.analyse(se_samples=20, report=True, write='results/weight_height.txt', add_note=True)
regress.evaluate()

#visualize results
plt.scatter(x,y)
plt.title('Female Height and Weight (5000 samples) inkl. Best Fit')
plt.ylabel('Height in inches')
plt.xlabel('Weight in lbs')
plt.plot(x, [regress.beta[0] + regress.beta[1]*xi for xi in x], color='red')
plt.savefig('results/weight_height_graph.png') 
plt.show()

#predict 
prediction = regress.predict(120)
print('Prediction: the person will weigh', prediction, 'lbs')






