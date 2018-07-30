import pandas as pd 
import numpy as np
import math 

df = pd.read_csv("test_scores.csv")

def gradient_descent(x,y):
	m_curr = b_curr = 0
	iterations = 1000000
	n = len(x)
	learning_rate = 0.000211
	pcost = 0

	for i in range(iterations):
		y_predicted = m_curr * x + b_curr
		cost = (1/n) * sum([val**2 for val in (y - y_predicted)]) 
		##using mse(mean squared error)
		md = -(2/n)*sum(x * (y - y_predicted))
		bd = -(2/n)*sum(y - y_predicted)
		m_curr = m_curr - learning_rate * md
		b_curr = b_curr - learning_rate * bd
		if(math.isclose(cost,pcost,rel_tol=1e-20)):
			print("m {}, b {}, cost {}, i {}".format(m_curr,b_curr,cost,i))
			break
		pcost = cost
		#print("m {}, b {}, cost {}, i {}".format(m_curr,b_curr,cost,i))
	#print("m {}, b {}, cost {}".format(m_curr,b_curr,cost))		

y = np.array(df.cs)
X = np.array(df.math)

gradient_descent(X,y)