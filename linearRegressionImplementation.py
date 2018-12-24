import numpy as np 
import matplotlib.pyplot as plt;
from matplotlib import style
from statistics import mean
style.use('fivethirtyeight')
def best_fit_slope(xs,ys):
    slope=(mean(xs)*mean(ys))
    slope=slope-(mean(xs*ys))
    slope=slope/((mean(xs)**2)-(mean(xs**2)))
    return slope
def get_yIntercept(ys,slope,xs):
    yIntercept=(mean(ys))-(slope*(mean(xs)))
    return yIntercept
xaxis = [1,2,3,4,5]
yaxis = [1,4,9,16,25]
xaxis = np.array(xaxis, dtype=np.float)
yaxis = np.array(yaxis, dtype=np.float)

slope=best_fit_slope(xaxis,yaxis)
yIntercept=get_yIntercept(yaxis,slope,xaxis)
print(slope, " ",yIntercept)

regression_line= [(slope*x)+yIntercept for x in xaxis]

plt.scatter(xaxis,yaxis)
plt.plot(xaxis,regression_line)
plt.title("REGRESSION LINE")
plt.show()