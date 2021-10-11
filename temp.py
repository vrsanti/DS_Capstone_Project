

# A first Python script
import sys
print(sys.platform)
print(2**100)
x = 'Spam!'
print(x * 2)

    


import math
math.pi
math.sqrt(85)

import random
random.Random()
random.choice([1,2,3,4])


S = 'Spam'
len(S)

S[0]

S[1]

S[-1]

S[-2]

S[len(S)-1]

S[1:4]

S = 'z' + S[1:]

S.find('pa')

S.replace('pa','XYZ')

S

line = 'aaa,bbb,ccc,dd'
line.split(',')

S.upper()

S.isalpha()

'%s, eggs, and %s' % ('spam','SPAM!')

'{0}, eggs, and {1}'.format('spam', 'SPAM!')


L = [123, 'spam', 1.23]
len(L)

L[0]

L[:-1]

L + [4, 5, 6]

L.append('NI')

L.pop(2)

L


M = ['bb','aa','cc']
M.sort()
M

M.reverse()
M



M = [[1, 2, 3],
     [4, 5, 6],
     [7, 8,9]]
M

M[1]

M[1][2]



col2 = [row[1] for row in M]
col2

[row[1] + 1 for row in M]

[row[1] for row in M if row[1] % 2 == 0]



import pandas as pd
wnba = pd.read_csv('/Users/santiagovasquez/Documents/Coding/WNBA Stats.csv')
parameter = wnba['Games Played'].max()
sample = wnba['Games Played'].sample(30, random_state = 1)
statistic = sample.max()
sampling_error = parameter - statistic


#Ejercicio 
import pandas as pd
import numpy as np
import sklearn as sklearn


df = pd.read_csv('/Users/santiagovasquez/Documents/Coding/imports-85.data', header= None)

df.head(5)
df.tail(5)

headers = ["symboling","normalized-losses","make","fuel-type",
           "aspiration","num-of-doors","body-style","drive-wheels",
           "engine-location","wheel-base","length","width","height",
           "curb-weight","engine-type","num-of-cylinders",
           "engine-size","fuel-system","bore","stroke",
           "compression-ratio","horsepower","peak-rpm","city-mpg"
           ,"highway-mpg","price"]

df.columns = headers

df.head(5)

path = "/Users/santiagovasquez/Documents/Coding/automobile.csv"

df.to_csv(path)

df.dtypes

df.describe(include="all")

df.info()

print("The last 10 rows of the dataframe\n")
df.tail(10)

df.replace('?', np.NaN, inplace = True)

df = df1.dropna(subset = ["normalized-losses"], axis = 0)   

df = df1

df.head(10)

# valida casos missing
missing_data = df.isnull()
missing_data.head(5)

# despliega casos missing en cada columna
for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())
    print("")   


avg_norm_loss = df["normalized-losses"].astype("float").mean(axis=0)
print("Average of normalized-losses:", avg_norm_loss)
df["normalized-losses"].replace(np.nan, avg_norm_loss, inplace=True)

avg_bore=df['bore'].astype('float').mean(axis=0)
print("Average of bore:", avg_bore)
df["bore"].replace(np.nan, avg_bore, inplace=True)

avg_stroke=df['stroke'].astype('float').mean(axis=0)
print("Average of stroke:", avg_stroke)
df["stroke"].replace(np.nan, avg_stroke, inplace=True)

df['num-of-doors'].value_counts()
df['num-of-doors'].value_counts().idxmax()
df["num-of-doors"].replace(np.nan, "four", inplace=True)


df.dropna(subset=["price"], axis=0, inplace=True)
df.reset_index(drop=True, inplace=True)

df.head()



df.dtypes

df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")
df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")
df[["price"]] = df[["price"]].astype("float")
df[["peak-rpm"]] = df[["peak-rpm"]].astype("float")


df[['fuel-type', 'fuel-system']].describe()

df["symboling"] = df["symboling"] + 1

mean = df["normalized-losses"].mean()

df["normalized-losses"].replace(np.nan,mean)

df["city-mpg"] = 235 / df["city-mpg"]
df.head(10)

df.rename(columns = {"city-mpg":"city_L/100km"}, inplace = True)    
df.head(10)

df["normalized-losses"].tail(5)
df = df1.dropna(subset = ["normalized-losses"], axis = 0)


df["normalized-losses"] = df["normalized-losses"].astype("int")


pd.get_dummies(df["fuel-type"])


df['length'] = df['length']/df['length'].max()
df['width'] = df['width']/df['width'].max()
df['height'] = df['height']/df['height'].max()


df[["length","width","height"]].head()


# binning data

avg_horsepower = df["horsepower"].astype("float").mean(axis=0)
print("Average of horsepower:", avg_horsepower)
df["horsepower"].replace(np.nan, avg_horsepower, inplace=True)

df["horsepower"]=df["horsepower"].astype(int, copy=True)



%matplotlib inline
import matplotlib as plt
from matplotlib import pyplot
plt.pyplot.hist(df["horsepower"])

# set x/y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")

bins = np.linspace(min(df["horsepower"]), max(df["horsepower"]), 4)
bins

group_names = ['Low', 'Medium', 'High']

df['horsepower-binned'] = pd.cut(df['horsepower'], bins, labels=group_names, include_lowest=True )
df[['horsepower','horsepower-binned']].head(20)

df["horsepower-binned"].value_counts()


%matplotlib inline
import matplotlib as plt
from matplotlib import pyplot
pyplot.bar(group_names, df["horsepower-binned"].value_counts())

# set x/y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")


# draw historgram of attribute "horsepower" with bins = 3
plt.pyplot.hist(df["horsepower"], bins = 3)

# set x/y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")



#Indicator Variable (or Dummy Variable)

df.columns

dummy_variable_1 = pd.get_dummies(df["fuel-type"])
dummy_variable_1.head()

dummy_variable_1.rename(columns={'gas':'fuel-type-gas', 'diesel':'fuel-type-diesel'}, inplace=True)
dummy_variable_1.head()

# merge data frame "df" and "dummy_variable_1" 
df = pd.concat([df, dummy_variable_1], axis=1)

# drop original column "fuel-type" from "df"
df.drop("fuel-type", axis = 1, inplace=True)

df.head()



# get indicator variables of aspiration and assign it to data frame "dummy_variable_2"
dummy_variable_2 = pd.get_dummies(df['aspiration'])

# change column names for clarity
dummy_variable_2.rename(columns={'std':'aspiration-std', 'turbo': 'aspiration-turbo'}, inplace=True)

# show first 5 instances of data frame "dummy_variable_1"
dummy_variable_2.head()


# merge the new dataframe to the original datafram
df = pd.concat([df, dummy_variable_2], axis=1)

# drop original column "aspiration" from "df"
df.drop('aspiration', axis = 1, inplace=True)

df.head()


df.to_csv('clean_df.csv')


#Ejercicio 
import pandas as pd
import numpy as np

%matplotlib inline
import matplotlib as plt
from matplotlib import pyplot

df = pd.read_csv('clean_df.csv')
df.head()


sns.boxplot(x = "drive-wheels", y = "price", data = df)


y = df["price"]
x = df["engine-size"]
plt.scatter(x,y)



# tablas agrupadas
df_test = df[['drive-wheels','body-style','price']]
df_grp = df_test.groupby(['drive-wheels','body-style'], as_index = False).mean()
df_grp

df[['price','body-style','drive-wheels']].groupby(['body-style','drive-wheels'], as_index = False).mean()


# tabla pivot
df_pivot = df_grp.pivot(index = 'drive-wheels', columns = 'body-style')
df_pivot


#heatmap
import matplotlib.pyplot as plt
plt.pcolor(df_pivot, cmap = 'RdBu')
plt.colorbar()
plt.show()


#
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.linear_model as LinearRegression






# list the data types for each column
print(df.dtypes)

df.corr()

df[['bore', 'stroke', 'compression-ratio', 'horsepower']].corr()


# Engine size as potential predictor variable of price
sns.regplot(x="engine-size", y="price", data=df)
plt.ylim(0,)


df[["engine-size", "price"]].corr()

sns.regplot(x="highway-mpg", y="price", data=df)

df[['highway-mpg','price']].corr()

sns.regplot(x = 'peak-rpm', y = 'price', data = df)

df[['stroke','price']].corr()

sns.regplot(x = 'stroke', y = 'price', data = df)

# variables categoricas

sns.boxplot(x = 'body-style', y = 'price', data = df)
sns.boxplot(x = 'engine-location', y = 'price', data = df)
sns.boxplot(x = 'drive-wheels', y = 'price', data = df)


# Descriptive Statistical Analysis

df.describe()

df.describe(include = ['object'])

df['drive-wheels'].value_counts()


df['drive-wheels'].value_counts().to_frame()

drive_wheels_counts = df['drive-wheels'].value_counts().to_frame()
drive_wheels_counts.rename(columns={'drive-wheels': 'value_counts'}, inplace=True)
drive_wheels_counts

drive_wheels_counts.index.name = 'drive-wheels'
drive_wheels_counts


# engine-location as variable
engine_loc_counts = df['engine-location'].value_counts().to_frame()
engine_loc_counts.rename(columns={'engine-location': 'value_counts'}, inplace=True)
engine_loc_counts.index.name = 'engine-location'
engine_loc_counts.head(10)


df['drive-wheels'].unique()

df_group_one = df[['drive-wheels','body-style','price']]


# grouping results
df_group_one = df_group_one.groupby(['drive-wheels'],as_index=False).mean()
df_group_one

# grouping results
df_gptest = df[['drive-wheels','body-style','price']]
grouped_test1 = df_gptest.groupby(['drive-wheels','body-style'],as_index=False).mean()
grouped_test1


grouped_pivot = grouped_test1.pivot(index='drive-wheels',columns='body-style')
grouped_pivot


df_gptest2 = df[['body-style','price']]
grouped_test_bodystyle = df_gptest2.groupby(['body-style'], as_index = False).mean()
grouped_test_bodystyle


import matplotlib.pyplot as plt
%matplotlib inline 


plt.pcolor(grouped_pivot, cmap = 'RdBu')
plt.colorbar()
plt.show()



fig, ax = plt.subplots()
im = ax.pcolor(grouped_pivot, cmap='RdBu')

#label names
row_labels = grouped_pivot.columns.levels[1]
col_labels = grouped_pivot.index

#move ticks and labels to the center
ax.set_xticks(np.arange(grouped_pivot.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(grouped_pivot.shape[0]) + 0.5, minor=False)

#insert labels
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)

#rotate label if too long
plt.xticks(rotation=90)

fig.colorbar(im)
plt.show()

path='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/automobileEDA.csv'
df = pd.read_csv(path)
df.head()

df.corr()


pearson_coef, p_value = stats.pearsonr(df['wheel-base'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)  

pearson_coef, p_value = stats.pearsonr(df['horsepower'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)  

pearson_coef, p_value = stats.pearsonr(df['length'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)  



# ANOVA: Analysis of Varianc

grouped_test2 = df_gptest[['drive-wheels', 'price']].groupby(['drive-wheels'])
grouped_test2.head(2)

df_gptest

grouped_test2.get_group('4wd')['price']

# ANOVA
f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['price'], grouped_test2.get_group('rwd')['price'], grouped_test2.get_group('4wd')['price'])  
 
print( "ANOVA results: F=", f_val, ", P =", p_val) 

# análisis separado

f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['price'], grouped_test2.get_group('rwd')['price'])  
 
print( "ANOVA results: F=", f_val, ", P =", p_val )


f_val, p_val = stats.f_oneway(grouped_test2.get_group('4wd')['price'], grouped_test2.get_group('rwd')['price'])  
   
print( "ANOVA results: F=", f_val, ", P =", p_val)   


f_val, p_val = stats.f_oneway(grouped_test2.get_group('4wd')['price'], grouped_test2.get_group('fwd')['price'])  
 
print("ANOVA results: F=", f_val, ", P =", p_val)   


from sklearn.linear_model import LinearRegression 

lm=LinearRegression()
X = df[['highway-mpg']]
Y = df['price']

lm.fit(X, Y)
Yhat = lm.predict(X)
Yhat


sns.residplot(df['highway-mpg'],df['price'])

axl = sns.displot(df['price'], hist = False, color = 'r', label = 'Actual value')
sns.distplot(Yhat, hist = False, color = 'b', label = 'Fitted values', ax = axl)



# Model development

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# path of data 
path = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/automobileEDA.csv'
df = pd.read_csv(path)
df.head()



#  Linear Regression and Multiple Linear Regression

from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm

X = df[['highway-mpg']]
Y = df['price']

lm.fit(X,Y)


# predicción
Yhat=lm.predict(X)
Yhat[0:5]   

# intercep and slope
lm.intercept_
lm.coef_

# 𝑌ℎ𝑎𝑡=𝑎+𝑏𝑋 Price = 38423.31 - 821.73 x highway-mpg

lm1 = LinearRegression()
lm1

lm1.fit(df[['engine-size']], df[['price']])
lm1

lm1.intercept_
lm1.coef_


#Multiple Linear Regression

Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]

lm.fit(Z, df['price'])
lm.intercept_
lm.coef_

lm2 = LinearRegression()
lm2.fit(df[['normalized-losses' , 'highway-mpg']],df['price'])
lm2.coef_


# model evaluation using visualization

import seaborn as sns
%matplotlib inline 

width = 12
height = 10

plt.figure(figsize=(width, height))
sns.regplot(x="highway-mpg", y="price", data=df)
plt.ylim(0,)

plt.figure(figsize=(width, height))
sns.regplot(x="peak-rpm", y="price", data=df)
plt.ylim(0,)

df[["peak-rpm","highway-mpg","price"]].corr()


width = 12
height = 10

plt.figure(figsize=(width, height))
sns.residplot(df['highway-mpg'], df['price'])
plt.show()



# multiple linear regression

Y_hat = lm.predict(Z)

plt.figure(figsize=(width, height))


ax1 = sns.distplot(df['price'], hist=False, color="r", label="Actual Value")
sns.distplot(Y_hat, hist=False, color="b", label="Fitted Values" , ax=ax1)


plt.title('Actual vs Fitted Values for Price')
plt.xlabel('Price (in dollars)')
plt.ylabel('Proportion of Cars')

plt.show()
plt.close()


# Polynomial regression and pipelines

def PlotPolly(model, independent_variable, dependent_variabble, Name):
    x_new = np.linspace(15, 55, 100)
    y_new = model(x_new)

    plt.plot(independent_variable, dependent_variabble, '.', x_new, y_new, '-')
    plt.title('Polynomial Fit with Matplotlib for Price ~ Length')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('Price of Cars')

    plt.show()
    plt.close()


x = df['highway-mpg']
y = df['price']

f = np.polyfit(x, y, 3)
p = np.poly1d(f)
print(p)

PlotPolly(p, x, y, 'highway-mpg')

np.polyfit(x, y, 3)




f1 = np.polyfit(x, y, 11)
p1 = np.poly1d(f1)
print(p1)
PlotPolly(p1,x,y, 'Highway MPG')




from sklearn.preprocessing import PolynomialFeatures

pr=PolynomialFeatures(degree=2)
pr

Z_pr=pr.fit_transform(Z)


Z.shape

Z_pr.shape



# Pipeline

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

Input=[('scale',StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), 
       ('model',LinearRegression())]

pipe=Pipeline(Input)
pipe

Z = Z.astype(float)
pipe.fit(Z,y)

ypipe=pipe.predict(Z)
ypipe[0:4]

# Measures for In-Sample Evaluation

#highway_mpg_fit
lm.fit(X, Y)
# Find the R^2
print('The R-square is: ', lm.score(X, Y))

Yhat=lm.predict(X)
print('The output of the first four predicted value is: ', Yhat[0:4])


from sklearn.metrics import mean_squared_error

mse = mean_squared_error(df['price'], Yhat)
print('The mean square error of price and predicted value is: ', mse)


# fit the model 
lm.fit(Z, df['price'])
# Find the R^2
print('The R-square is: ', lm.score(Z, df['price']))

Y_predict_multifit = lm.predict(Z)

print('The mean square error of price and predicted value using multifit is: ', \
      mean_squared_error(df['price'], Y_predict_multifit))


from sklearn.metrics import r2_score

r_squared = r2_score(y, p(x))
print('The R-square value is: ', r_squared)



mean_squared_error(df['price'], p(x))


# Prediction and Decision Making

import matplotlib.pyplot as plt
import numpy as np

%matplotlib inline 

new_input=np.arange(1, 100, 1).reshape(-1, 1)
lm.fit(X, Y)
lm


yhat=lm.predict(new_input)
yhat[0:5]


plt.plot(new_input, yhat)
plt.show()





"hello Mike".find("Mike") 

str(1+1)


A = (1,2,3,4,5)
len(A)
A[1:4]




B=[1,2,[3,'a'],[4,'b']]
B[3][1]



# While Loop Example

dates = [1982, 1980, 1973, 2000]

i = 0
year = dates[0]

while(year != 1973):    
    print(year)
    i = i + 1
    year = dates[i]
    

print("It took ", i ,"repetitions to get out of loop.")





def add1(a):
    b = a + 1;
    print(a, 'plus 1 equals ', b)
    return b

add1(2)


def printStuff(Stuff):
    for i,s in enumerate(Stuff):
        print('Album', i, 'Rating is ', s)

album_ratings = [10.0, 8.5, 9.5]
printStuff(album_ratings)




a = 1

try:
    b = int(input("Please enter a number to divide a"))
    a = a/b
    print("Success a=",a)
except:
    print("There was an error")



a = 1

try:
    b = int(input("Please enter a number to divide a"))
    a = a/b
except ZeroDivisionError:
    print("The number you provided cant divide 1 because it is 0")
except ValueError:
    print("You did not provide a number")
except:
    print("Something went wrong")
else:
    print("success a=",a)
finally:
    print("Processing Complete")



RecCircle = circle(10,'red')



import matplotlib.pyplot as plt
%matplotlib inline  


# Create a class Circle

class Circle(object):
    
    # Constructor
    def __init__(self, radius=3, color='blue'):
        self.radius = radius
        self.color = color 
    
    # Method
    def add_radius(self, r):
        self.radius = self.radius + r
        return(self.radius)
    
    # Method
    def drawCircle(self):
        plt.gca().add_patch(plt.Circle((0, 0), radius=self.radius, fc=self.color))
        plt.axis('scaled')
        plt.show()  


RedCircle = Circle(10, 'red')


dir(RedCircle)

RedCircle.radius

RedCircle.color

RedCircle.radius = 1
RedCircle.radius

RedCircle.drawCircle()


# Use method to change the object attribute radius

print('Radius of object:',RedCircle.radius)
RedCircle.add_radius(2)
print('Radius of object of after applying the method add_radius(2):',RedCircle.radius)
RedCircle.add_radius(5)
print('Radius of object of after applying the method add_radius(5):',RedCircle.radius)


BlueCircle = Circle(radius=100)

BlueCircle.radius
BlueCircle.color
BlueCircle.drawCircle()


# Create a new Rectangle class for creating a rectangle object

class Rectangle(object):
    
    # Constructor
    def __init__(self, width=2, height=3, color='r'):
        self.height = height 
        self.width = width
        self.color = color
    
    # Method
    def drawRectangle(self):
        plt.gca().add_patch(plt.Rectangle((0, 0), self.width, self.height ,fc=self.color))
        plt.axis('scaled')
        plt.show()


SkinnyBlueRectangle = Rectangle(2, 10, 'blue')

SkinnyBlueRectangle.drawRectangle()






class analysedText(object):
    
    def __init__ (self, text):
        # remove punctuation
        formattedText = text.replace('.','').replace('!','').replace('?','').replace(',','')
        
        # make text lowercase
        formattedText = formattedText.lower()
        
        self.fmtText = formattedText
        
    def freqAll(self):        
        # split text into words
        wordList = self.fmtText.split(' ')
        
        # Create dictionary
        freqMap = {}
        for word in set(wordList): # use set to remove duplicates in list
            freqMap[word] = wordList.count(word)
        
        return freqMap
    
    def freqOf(self,word):
        # get frequency map
        freqDict = self.freqAll()
        
        if word in freqDict:
            return freqDict[word]
        else:
            return 0



class Points(object):
  def __init__(self,x,y):

    self.x=x
    self.y=y

  def print_point(self):

    print('x=',self.x,' y=',self.y)

p1=Points(1,2)
p1.print_point()


x=0
while(x<2):
    print(x)
    x=x+1  


for i,x in enumerate(['A','B','C']):
    print(i+1,x)



class Points(object):

  def __init__(self,x,y):

    self.x=x
    self.y=y

  def print_point(self):

    print('x=',self.x,' y=',self.y)

p2=Points(1,2)

p2.x='A'

p2.print_point()


def step(x):
    if x>0:
        y=1
    else:
        y=0
    return y



a=1

def do(x):
    return(x+a)

print(do(1))



x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)

plt.plot(x,y)



a=np.array([0,1])
b=np.array([1,0])
np.dot(a,b) 

X=np.array([[1,0],[0,1]])
Y=np.array([[0,1],[1,0]])
Z=X+Y
Z


X=np.array([[1,0,1],[2,2,2]]) 
out=X[0,1:3]
out

X

X=np.array([[1,0],[0,1]])
Y=np.array([[2,2],[2,2]])
Z=np.dot(X,Y)
Z



import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.offline import plot
import matplotlib.pyplot as plt
import datetime
from pycoingecko import CoinGeckoAPI
from mplfinance.original_flavor import candlestick2_ohlc


name = 'Lizz',
print(name[0:2])



var = '01234567'
print(var[::2])
Find



import matplotlib.pyplot as plt 
transparency = 0.35 
area_df.plot(kind='area', alpha=transparency, figsize=(20, 10)) 
plt.title('Plot Title') 
plt.ylabel('Vertical Axis Label')
plt.xlabel('Horizontal Axis Label') 
plt.show() 


url = " https://www.macrotrends.net/stocks/charts/TSLA/tesla/revenue"
html_data  = requests.get(url).text



!pip install yfinance
#!pip install pandas
#!pip install requests
!pip install bs4
#!pip install plotly
import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
import plotly.graph_objects as go
from plotly.subplots import make_subplots


url = " https://www.macrotrends.net/stocks/charts/TSLA/tesla/revenue"
html_data  = requests.get(url).text

soup = BeautifulSoup(html_data, 'html5lib')


tesla_revenue = pd.DataFrame(columns=["Date", "Revenue"])

for row in soup.find("tbody").find_all("tr"):
    col = row.find_all("td")
    Date = col[0].text
    Revenue = col[1].text
    tesla_revenue = tesla_revenue.append({'Date':Date, 'Revenue':Revenue}, ignore_index = True)

tesla_revenue


for i in soup.find_all('table'):
    print(i)
    
    
print(soup)
















