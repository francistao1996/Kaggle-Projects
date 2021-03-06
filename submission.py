
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib qt5
import math
import datetime
import xgboost as xgb
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

# Import dataset
bike = pd.read_csv('D:/Udemy/Data Science 2020 Complete Data Science and Machine Learning/006 - Kaggle Project/train.csv')

# Copy the dataset and drop the irrelavant columns
bikec = bike.copy()
bikec = bikec.drop(['casual','registered'], axis = 1)

# Check missing values
bikec.isnull().sum()

# Vusualize data by histogram
bikec.hist(rwidth=0.9)
plt.tight_layout()

# Create Year, Month, Hour 
bikec['year'] = pd.DatetimeIndex(bikec['datetime']).year.map({2011:0,2012:1})
bikec['month'] = pd.DatetimeIndex(bikec['datetime']).month
bikec['hour'] = pd.DatetimeIndex(bikec['datetime']).hour

# Visualize the continuous data vs demand

plt.subplot(2,2,1)
plt.title('Temperature VS Count')
plt.scatter(bikec['temp'], bikec['count'], s=2, c = 'g')

plt.subplot(2,2,2)
plt.title('aTemperature VS Count')
plt.scatter(bikec['atemp'], bikec['count'], s=2, c = 'r')

plt.subplot(2,2,3)
plt.title('Humidity VS Count')
plt.scatter(bikec['humidity'], bikec['count'], s=2,c = 'b')

plt.subplot(2,2,4)
plt.title('Windspeed VS Count')
plt.scatter(bikec['windspeed'], bikec['count'], s=2,c = 'c')

plt.tight_layout()

# Visualize the categorical data vs count
# Create a 3x3 subplot
# Create a list of unique season's values
# Create average Demand for unique seasons
plt.subplot(3,3,1)
plt.title('Average count per Season')
season_uni = bikec['season'].unique()
demands_avg = bikec.groupby('season')['count'].mean()
plt.bar(season_uni, demands_avg, color = ['b','r','y','m'])

# Create a list of unique year's values
# Create average Demand for unique years
plt.subplot(3,3,2)
plt.title('Average count per Year')
year_uni = bikec['year'].unique()
demandy_avg = bikec.groupby('year')['count'].mean()
plt.bar(year_uni, demandy_avg, color = ['b','r','y','m'])

# Create a list of unique month's values
# Create average Demand for unique months
plt.subplot(3,3,3)
plt.title('Average count per Month')
month_uni = bikec['month'].unique()
demandm_avg = bikec.groupby('month')['count'].mean()
plt.bar(month_uni, demandm_avg, color = ['b','r','y','m'])

# Create a list of unique hour's values
# Create average Demand for unique hours
plt.subplot(3,3,4)
plt.title('Average count per Hour')
hour_uni = bikec['hour'].unique()
demandhr_avg = bikec.groupby('hour')['count'].mean()
plt.bar(hour_uni, demandhr_avg, color = ['b','r','y','m'])

# Create a list of unique holiday's values
# Create average Demand for unique holidays
plt.subplot(3,3,5)
plt.title('Average count per Holiday')
holiday_uni = bikec['holiday'].unique()
demandho_avg = bikec.groupby('holiday')['count'].mean()
plt.bar(holiday_uni, demandho_avg, color = ['b','r','y','m'])


# Create a list of unique workingday's values
# Create average Demand for unique workingdays
plt.subplot(3,3,6)
plt.title('Average count per Workingday')
workingday_uni = bikec['workingday'].unique()
demandwork_avg = bikec.groupby('workingday')['count'].mean()
plt.bar(workingday_uni, demandwork_avg, color = ['b','r','y','m'])

# Create a list of unique weather's values
# Create average Demand for unique weathers
plt.subplot(3,3,7)
plt.title('Average count per Weather')
weather_uni = bikec['weather'].unique()
demandwea_avg = bikec.groupby('weather')['count'].mean()
plt.bar(weather_uni, demandwea_avg, color = ['b','r','y','m'])

plt.tight_layout()

# Drop features - Year, Workingday

# Check for outliers
bikec['count'].describe()

bikec['count'].quantile([0.05, 0.1, 0.15, 0.9, 0.95, 0.99])

bikec = bikec[np.abs(bikec['count']-bikec['count'].mean()) <= (3*bikec['count'].std())]
# Check multiple linear regression assumptions
# Create the correlation matrix
correlation = bikec[['temp','atemp','humidity','windspeed','count']].corr()

# Drop features above and added 'windspeed' cuz its correlation is too low with demand
bikec_remain = bikec

# Check the autocorrelation in demand using acorr
df1 = pd.to_numeric(bikec_remain['count'], downcast = 'float')
plt.acorr(df1, maxlags=14)

# There is high autocorrelation in count
# Count is not normally distributed
# So we have to take log of the feature 'count' in order to normalize it
df1 = bikec_remain['count']
df2 = np.log(df1)

plt.figure()
df1.hist(rwidth=0.9, bins = 20)

plt.figure()
df2.hist(rwidth=0.9, bins = 20)

# Change the demand into logarithmic form
bikec_remain['count'] = np.log(bikec_remain['count'])

bikec_remain.dtypes

bikec_remain['year'] = bikec_remain['year'].astype('category')
bikec_remain['season'] = bikec_remain['season'].astype('category')
bikec_remain['month'] = bikec_remain['month'].astype('category')
bikec_remain['hour'] = bikec_remain['hour'].astype('category')
bikec_remain['holiday'] = bikec_remain['holiday'].astype('category')
bikec_remain['weather'] = bikec_remain['weather'].astype('category')
# Drop datetime first
bikec_remain1 = bikec_remain.drop(['datetime'], axis=1)

bike_dummy = pd.get_dummies(bikec_remain1, drop_first = True)

Y = bike_dummy[['count']]
X = bike_dummy.drop(['count'], axis=1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = \
  train_test_split(X, Y, test_size = 0.25, random_state = 50)

# Prediction
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

# Create regressors
xgbr = xgb.XGBRegressor(colsample_bytree=0.45, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2000,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, random_state =7, nthread = -1)
regg = GradientBoostingRegressor(n_estimators = 2000)
regf = RandomForestRegressor()
regf2 = RandomForestRegressor(n_estimators=500)
std_reg = LinearRegression()
gsvf = GridSearchCV(estimator=regf2,param_grid={'n_estimators':[500],'n_jobs':[-1],'max_features':["auto",'sqrt','log2']},scoring='neg_mean_squared_log_error')


# Train models
gsvf.fit(x_train, y_train)
xgbr.fit(x_train,y_train)
regg.fit(x_train, y_train)
regf2.fit(x_train, y_train)
regf.fit(x_train, y_train)
std_reg.fit(x_train, y_train)

r2_trf = regf.score(x_train, y_train)
r2_trf = regf.score(x_test, y_test)
r2_tr = std_reg.score(x_train, y_train)
r2_te = std_reg.score(x_test, y_test)

y_testexp = np.exp(y_test)
# Predicting Y anc calculating RMSE

Y_predgsvf = gsvf.predict(x_test)
Y_predxgb = xgbr.predict(x_test)
Y_predg = regg.predict(x_test)
Y_predf = regf.predict(x_test)
Y_pred = std_reg.predict(x_test)
Y_predf2 = regf2.predict(x_test)

from sklearn.metrics import mean_squared_error
rmsef = math.sqrt(mean_squared_error(y_test, Y_predf))
rmse = math.sqrt(mean_squared_error(y_test, Y_pred))
rmsef2 = math.sqrt(mean_squared_error(y_test, Y_predf2))
rmseg = math.sqrt(mean_squared_error(y_test, Y_predg))
rmsex = math.sqrt(mean_squared_error(y_test, Y_predxgb))
rmsegsvf = math.sqrt(mean_squared_error(y_test, Y_predgsvf))


# Calculating RMSLE
from sklearn.metrics import mean_squared_log_error
rmslef = math.sqrt(mean_squared_log_error(y_testexp, np.exp(Y_predf)))
#rmsle = math.sqrt(mean_squared_log_error(y_test, Y_pred))
rmslef2 = math.sqrt(mean_squared_log_error(y_testexp, np.exp(Y_predf2)))
rmsleg = math.sqrt(mean_squared_log_error(y_testexp,np.exp(Y_predg)))
rmslex = math.sqrt(mean_squared_log_error(y_testexp, np.exp(Y_predxgb)))
rmslegsvf = math.sqrt(mean_squared_log_error(y_testexp,np.exp(Y_predgsvf)))

# Import test data
testdata = pd.read_csv('D:/Udemy/Data Science 2020 Complete Data Science and Machine Learning/006 - Kaggle Project/test.csv')

time = pd.DataFrame(testdata['datetime'])

testdata2 = testdata.drop(['datetime'], axis = 1)

# Extract month and hour
testdata2['year'] = pd.DatetimeIndex(time['datetime']).year.map({2011:0,2012:1})
testdata2['month'] = pd.DatetimeIndex(time['datetime']).month
testdata2['hour'] = pd.DatetimeIndex(time['datetime']).hour

# Reorder the columns to match the train set
# testdata2 = testdata2[['season','month','hour','holiday','weather','temp','humidity']]

# Check data types and distribution

testdata2.dtypes
# Convert to category
testdata2['year'] = testdata2['year'].astype('category')
testdata2['season'] = testdata2['season'].astype('category')
testdata2['month'] = testdata2['month'].astype('category')
testdata2['hour'] = testdata2['hour'].astype('category')
testdata2['holiday'] = testdata2['holiday'].astype('category')
testdata2['weather'] = testdata2['weather'].astype('category')

testdata_dummy = pd.get_dummies(testdata2, drop_first = True)

Y_pred2 = pd.DataFrame(std_reg.predict(testdata_dummy))
Y_pred3 = std_reg.predict(testdata_dummy)
Y_pred4 = regf.predict(testdata_dummy)
Y_pred5 = regf2.predict(testdata_dummy)
Y_pred6 = regg.predict(testdata_dummy)
Y_pred7 = xgbr.predict(testdata_dummy)
Y_predgsvf = gsvf.predict(testdata_dummy)



predgsvf = np.exp(Y_predgsvf) #gsvf
pred7 = np.exp(Y_pred7) #xgbr
pred6 = np.exp(Y_pred6) #regg
pred4 = np.exp(Y_pred4) #regf
pred3 = np.exp(Y_pred3)
pred5 = np.exp(Y_pred5) # regf2


ans4 = pd.concat([time,Y_pred2], axis=1)
ans4.columns = ['datetime','count']
ans4.to_csv('D:/Kaggle/answer4.csv', index=False)

ans5 = pd.concat([time,pd.DataFrame(pred3)], axis=1)
ans5.columns = ['datetime','count']
ans5.to_csv('D:/Kaggle/answer5.csv', index=False)

ans6 = pd.concat([time,pd.DataFrame(pred4)], axis=1)
ans6.columns = ['datetime','count']
ans6.to_csv('D:/Kaggle/answer6.csv', index=False)

ans7 = pd.concat([time,pd.DataFrame(pred5)], axis=1)
ans7.columns = ['datetime','count']
ans7.to_csv('D:/Kaggle/answer7.csv', index=False)

ans8 = pd.concat([time,pd.DataFrame(pred6)], axis=1)
ans8.columns = ['datetime','count']
ans8.to_csv('D:/Kaggle/answer8.csv', index=False)

ans9 = pd.concat([time,pd.DataFrame(pred7)], axis=1)
ans9.columns = ['datetime','count']
ans9.to_csv('D:/Kaggle/answer9.csv', index=False)

ans10 = pd.concat([time,pd.DataFrame(predgsvf)], axis=1)
ans10.columns = ['datetime','count']
ans10.to_csv('D:/Kaggle/answer10.csv', index=False)

