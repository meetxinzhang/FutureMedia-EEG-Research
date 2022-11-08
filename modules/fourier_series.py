# encoding: utf-8
"""
 @author: Xin Zhang
 @contact: 2250271011@email.szu.edu.cn
 @time: 2022/11/4 16:09
 @name: 
 @desc:
"""

import numpy as np
from scipy.optimize import curve_fit as curve_fit
from symfit import parameters, variables, sin, cos, Fit
import matplotlib.pyplot as plt

tau = np.pi * 4
x = np.arange(0, 10, 0.001)
y = 3 * np.sin(x) + np.cos(5 * x + 0.2)


# def fourier(x, *popt):
#     ret = popt[0]
#     a = popt[1:5]
#     b = popt[5:]
#     for deg in range(0, len(a)):
#         ret += a[deg] * np.cos(deg * np.pi/tau * x) + b[deg] * np.sin(deg * np.pi/tau * x)
#     return ret
#
#
# # Fit with 15 harmonics
# popt, pcov = curve_fit(f=fourier, xdata=x, ydata=y, p0=[1.0] * 9)
# print(popt)
#
# # Plot data, 15 harmonics, and first 3 harmonics
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# p1, = plt.plot(x, y)
# p2, = plt.plot(x, fourier(x, *popt))
# # p3, = plt.plot(x, fourier(x, popt[0], popt[1], popt[2]))
# plt.show()

def fourier_series(x, f, n=0):
    """
    Returns a symbolic fourier series of order `n`.
    @param n: Order of the fourier series.
    @param x: Independent variable
    @param f: Frequency of the fourier series
    """
    # Make the parameter objects for all the terms
    a0, *cos_a = parameters(','.join(['a{}'.format(i) for i in range(0, n + 1)]))
    sin_b = parameters(','.join(['b{}'.format(i) for i in range(1, n + 1)]))
    # Construct the series
    series = a0 + sum(ai * cos(i * f * x) + bi * sin(i * f * x)
                      for i, (ai, bi) in enumerate(zip(cos_a, sin_b), start=1))
    return series


xv, yv = variables('x, y')
w, = parameters('w')
model_dict = {yv: fourier_series(xv, f=w, n=4)}
print(model_dict)


# Make step function data
# Define a Fit object for this model and data
fit = Fit(model_dict, x=x, y=y)
fit_result = fit.execute()
print(fit_result)

# Plot the result
plt.plot(x, y)
plt.plot(x, fit.model(x=x, **fit_result.params).y, color='green', ls=':')
plt.show()
