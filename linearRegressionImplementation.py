import numpy as np 
import matplotlib.pyplot as plt;
from matplotlib import style
from statistics import mean
style.use('ggplot')
def best_fit_slope(xs,ys):
    slope=(mean(xs)*mean(ys))
    slope=slope-(mean(xs*ys))
    slope=slope/((mean(xs)**2)-(mean(xs**2)))
    return slope

def get_yIntercept(ys,slope,xs):
    yIntercept=(mean(ys))-(slope*(mean(xs)))
    return yIntercept

def get_r_squared_error(regression_line,yaxis):
    meanArray=np.array([mean(yaxis) for y in yaxis],dtype=np.float)
    r_squared=sum((regression_line-yaxis)**2)
    r_squaredd=sum((meanArray-yaxis)**2)
    r_squared=1-(r_squared/r_squaredd)
    return r_squared
xaxis = [1,2,3,4,5]
yaxis = [1,4,9,16,25]
xaxis = np.array(xaxis, dtype=np.float)
yaxis = np.array(yaxis, dtype=np.float)

slope=best_fit_slope(xaxis,yaxis)
yIntercept=get_yIntercept(yaxis,slope,xaxis)
print(slope, " ",yIntercept)

regression_line= [(slope*x)+yIntercept for x in xaxis]
regression_line_array=np.array(regression_line,dtype=np.float)

error=get_r_squared_error(regression_line,yaxis)
print(error)
#sample prediction
x=6
y=slope*x + yIntercept


print(y)


plt.scatter(xaxis,yaxis)
plt.plot(xaxis,regression_line)
plt.scatter(x,y,color='b')
plt.title("REGRESSION LINE")
plt.show()


#calculate the accuracy - r squared error r=1-(regression_line_squared_error - squared error mean y)

