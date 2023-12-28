#!/usr/bin/env python
# coding: utf-8

# In Machine Learning, polynomial regression is an algorithm that allows us to model nonlinear relationships between input features and output labels. It can be used in real-time business problems, such as sales forecasting, where the relationship between variables is not linear. Let’s understand how the Polynomial Regression algorithm works by taking an example of a real-time business problem.
# 
# Suppose you work as a Data Science professional in a company that sells a certain product. You have historical sales data from past years and want to predict next year’s sales. However, the relationship between sales and time (in months) is not linear, and you cannot use a simple linear regression model to accurately predict future sales.
# 
# This is where polynomial regression comes in. Instead of using a straight line to fit the data, it fits a polynomial curve of degree ‘n’ to the data points. The degree ‘n’ determines the complexity of the curve and can be chosen according to the degree of non-linearity of the data. For example, if the data has a quadratic relationship, we can use a degree of 2, which will fit a parabolic curve to the data points.

# # Advantages and Disadvantages of Polynomial Regression Algorithm
# Advantages:
# 
# Polynomial regression can model a wide range of nonlinear relationships between input and output variables. It can capture complex patterns that are difficult to model with linear regression.
# Polynomial regression is a simple algorithm that can be easily implemented and understood. It does not require advanced mathematical knowledge or complex algorithms.
# 
# Disadvantages:
# 
# Polynomial regression can easily overfit the data if the degree of the polynomial curve is too high. It can lead to poor generalization and inaccurate predictions on new data.
# Polynomial regression can be sensitive to outliers in the data. Outliers can significantly affect the shape of the polynomial curve and lead to inaccurate predictions.
# Summary

# In[7]:


import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# create a imput
month= np.array([1,2,3,4,5,6,7,8,9,10])
sale =np.array([10,20,30,40,150,60,270,180,120,300])


# In[ ]:





# In[8]:


poly_reg =PolynomialFeatures(degree =4)
x_ploy =poly_reg.fit_transform(month.reshape(-1,1))
lin_reg =LinearRegression()
lin_reg.fit(x_ploy,sale)


# In[11]:


future_month =np.array([11,12,13])
future_x_ploy =poly_reg.fit_transform(future_month.reshape(-1,1))
future_sale =lin_reg.predict(future_x_ploy)
print(future_sale)


# In[14]:


fig = go.Figure()
fig.add_trace(go.Scatter(x=month, y=sale, name='Actual Sales'))
fig.add_trace(go.Scatter(x=month, y=lin_reg.predict(x_ploy), name='Fitted Curve'))
fig.add_trace(go.Scatter(x=future_month, y=future_sale, name='Predicted Sales'))
fig.show()


# In[ ]:




