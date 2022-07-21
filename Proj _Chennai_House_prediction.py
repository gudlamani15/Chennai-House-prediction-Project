#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[5]:


df = pd.read_csv("train-chennai-sale.csv")


# In[6]:


df.head(5)


# In[7]:


df = df.drop(columns="PRT_ID")


# In[8]:


df


# # 1 AREA

# In[9]:


df["AREA"].isnull().sum()


# In[10]:


df.AREA.values


# In[11]:


df.AREA.unique()


# In[12]:


df["AREA"] = df["AREA"].replace({"Adyr":"Adyar","Ann Nagar":"Anna Nagar","Ana Nagar":"Anna Nagar","Chrmpet":"Chormpet","Chrompt":"Chormpet","Chrompet":"Chormpet","KKNagar":"KK Nagar","Karapakam":"Karapakkam","TNagar":"T Nagar","Velachery":"Velchery"})


# In[13]:


df.AREA.unique()


# In[14]:


df.AREA.value_counts().plot.bar()
plt.xlabel("Area")
plt.ylabel("count")
plt.show


# In[15]:


#From plot we can see chormpet having more house sale
#T Nagar having least house sale


# In[16]:


df.groupby("AREA")["SALES_PRICE"].mean().plot.bar()


# In[17]:


df.groupby("AREA")["SALES_PRICE"].mean().sort_values(ascending = True).plot.bar()


# In[18]:


#FRom graph we can say that average price of house in Tnagar costly compare to other areas and low cost on karapakkam 
#the avergae of house is linearly increasing 
#so label encoding is preferable for Area column since we can see ordinal behaviour.


# In[19]:


Area = {"Karapakkam":0,"Adyar":1,"Chormpet":2,"Velchery":3,"KK Nagar":4,"Anna Nagar":5,"T Nagar":6}


# In[20]:


df.AREA = df["AREA"].replace(Area)


# In[21]:


df.columns


# # 2 INT_SQFT

# In[22]:


df.INT_SQFT.values


# In[23]:


df.INT_SQFT.isnull().sum()


# In[24]:


df.INT_SQFT.describe()


# In[25]:


q3 = df.INT_SQFT.quantile(0.75)
q1 = df.INT_SQFT.quantile(0.25)
iqr = q3 - q1 
iqr


# In[26]:


q3 + 1.5*iqr


# In[27]:


q1 - 1.5*iqr


# In[28]:


df.INT_SQFT.dtype


# In[29]:


# there is no outliers and null values


# In[30]:


df.INT_SQFT.plot.hist(bins=20)


# In[31]:


df.plot.scatter(x="INT_SQFT",y="SALES_PRICE")


# In[32]:


#from the graph we can see the there is good linear relation between int_sqft and sales_price


# In[33]:


df[["INT_SQFT","SALES_PRICE"]].corr()


# # 3 DATE_SALE

# In[34]:


df.DATE_SALE.values


# In[35]:


df.DATE_SALE.dtype


# In[36]:


df.DATE_SALE = pd.to_datetime(df.DATE_SALE)


# In[37]:


df.DATE_SALE.dtype


# In[38]:


df["Sale_year"] = pd.DatetimeIndex(df['DATE_SALE']).year


# In[39]:


df.info()


# In[40]:


df.DATE_SALE.isnull().sum()


# #  4 DIST_MAINROAD

# In[41]:


df.DIST_MAINROAD.values


# In[42]:


df.DIST_MAINROAD.isnull().sum()


# In[43]:


df.DIST_MAINROAD.dtype


# In[44]:


df.DIST_MAINROAD.describe()


# In[45]:


q3 = df.DIST_MAINROAD.quantile(0.75)
q1 = df.DIST_MAINROAD.quantile(0.25)
iqr = q3 - q1 
iqr


# In[46]:


q3 + 1.5*iqr


# In[47]:


q1 - 1.5*iqr


# In[48]:


df.DIST_MAINROAD.plot.hist(bins=50)


# In[49]:


df.plot.scatter(x="DIST_MAINROAD",y="SALES_PRICE")


# In[50]:


df["DIST_MAINROAD_sqr"] = df.DIST_MAINROAD**2


# In[51]:


df.plot.scatter(x="DIST_MAINROAD_sqr",y="SALES_PRICE")


# In[52]:


df["Transformed"]= np.log(df.DIST_MAINROAD)
df.Transformed.values


# In[53]:


df.plot.scatter(x="Transformed",y="SALES_PRICE")


# In[54]:


#we don't see any relationship between dist_mainroad and sale_price we dropping the dist_mairoad column


# In[55]:


#No outliers


# In[56]:


df =df.drop(columns=["Transformed","DIST_MAINROAD_sqr","DIST_MAINROAD"])


# # 5 N_BEDROOM

# In[57]:


df.N_BEDROOM.values


# In[58]:


df.N_BEDROOM.value_counts()


# In[59]:


df.N_BEDROOM.isnull().sum()


# In[60]:


x = df.N_BEDROOM.mode()
df.N_BEDROOM.fillna(int(x),inplace = True)


# In[61]:


df.N_BEDROOM.isnull().sum()


# In[62]:


df.N_BEDROOM.value_counts()


# In[63]:


df.N_BEDROOM.value_counts().plot.bar()
plt.title('beedroom vs count')
plt.xlabel("beedroom")
plt.ylabel("count")
plt.show


# In[64]:


df.groupby("N_BEDROOM")["SALES_PRICE"].mean().plot.bar()
plt.title('beedroom vs sale_price')
plt.xlabel("beedroom")
plt.ylabel("price")
plt.show


# In[65]:


# we can say that maximun houses are having 1 bedroom
#we can see that the average of 4 bed room house is high and we can see a linear relation between bedroom and sale price.


# In[66]:


df.columns


# # 5 N_BATHROOM

# In[67]:


df.N_BATHROOM.values


# In[68]:


df.N_BATHROOM.value_counts()


# In[69]:


df.N_BATHROOM.isnull().sum()


# In[70]:


df.N_BATHROOM.isnull().sum()*100/df.shape[0]


# In[71]:


x = df.N_BATHROOM.mode()


# In[72]:


df.N_BATHROOM.fillna(int(x),inplace = True)


# In[73]:


df.N_BATHROOM.isnull().sum()


# In[74]:


df.N_BATHROOM.value_counts().plot.bar()
plt.title('bathroom vs count')
plt.xlabel("bathroom")
plt.ylabel("count")
plt.show


# In[75]:


df.groupby("N_BATHROOM")["SALES_PRICE"].mean().plot.bar()
plt.title('bathroom vs sale_price')
plt.xlabel("bathroom")
plt.ylabel("price")
plt.show


# In[76]:


#we can see a linear relation between bathroom and saleprice


# # 6 N_ROOM

# In[77]:


df.N_ROOM.values


# In[78]:


df.N_ROOM.value_counts()


# In[79]:


df.N_ROOM.isnull().sum()


# In[80]:


df.N_ROOM.value_counts().plot.bar()
plt.title('room vs count')
plt.xlabel("room")
plt.ylabel("count")
plt.show


# In[81]:


# from graph we can say the demand of 4 room house is more and 6 and 2 room houses are less


# In[82]:


df.groupby("N_ROOM")["SALES_PRICE"].mean().plot.bar()
plt.title('room vs sale_price')
plt.xlabel("room")
plt.ylabel("price")
plt.show


# In[83]:


#there is linear relkation between room and saleprice


# # 7 SALE_COND

# In[84]:


df.SALE_COND.values


# In[85]:


df.SALE_COND.unique()


# In[86]:


df["SALE_COND"] = df["SALE_COND"].replace({"AdjLand":"Adj Land","Partiall":"Partial","PartiaLl":"Partial","AbNormal":"Ab Normal"})


# In[87]:


df.SALE_COND.unique()


# In[88]:


df.SALE_COND.value_counts()


# In[89]:


df.SALE_COND.isnull().sum()


# In[90]:


df.SALE_COND.value_counts().plot.bar()
plt.xlabel("salecond")
plt.ylabel("count")
plt.show


# In[91]:


df.groupby("SALE_COND")["SALES_PRICE"].mean().sort_values().plot.bar()
plt.title('sale_cond vs sale_price')
plt.xlabel("salecond")
plt.ylabel("price")
plt.show


# In[92]:


#from graph we can see that there is slight linearship between sale cond and sale price


# In[93]:


Sale_cond = {"Partial":0,"Family":1,"Ab Normal":2,"Normal Sale":3,"Adj Land":4}


# In[94]:


df.SALE_COND =df.SALE_COND.replace(Sale_cond)
    


# In[95]:


df.shape


# In[96]:


df.columns


# # 8 PARK_FACIL

# In[97]:


df.PARK_FACIL.values


# In[98]:


df.PARK_FACIL.isnull().sum()


# In[99]:


df.PARK_FACIL = df.PARK_FACIL.replace({"Yes":1,"No":0,"Noo":0})


# In[100]:


df.PARK_FACIL.values


# In[101]:


df.PARK_FACIL.value_counts().plot.bar()


# In[102]:


df.groupby("PARK_FACIL")["SALES_PRICE"].mean().plot.bar()


# In[103]:


#we can see a linear relation between parkfacil and salesprice


# # 9 DATE_BUILD

# In[104]:


df.DATE_BUILD.values


# In[105]:


df.DATE_BUILD.dtype


# In[106]:


df.DATE_BUILD.isnull().sum()


# In[107]:


df.DATE_BUILD = pd.to_datetime(df.DATE_BUILD)


# In[108]:


df.DATE_BUILD.dtype


# In[109]:


df["Build_year"] = pd.DatetimeIndex(df['DATE_BUILD']).year


# In[110]:


df.shape


# In[111]:


df["AGE"] = df["Sale_year"] - df["Build_year"]


# In[112]:


df["AGE"]


# # 10 AGE

# In[113]:


df.AGE.values


# In[114]:


df.AGE.plot.hist(bins=10)


# In[115]:


df.AGE.describe()


# In[116]:


q3 = df.AGE.quantile(0.75)
q1 = df.AGE.quantile(0.25)
iqr = q3-q1
iqr


# In[117]:


q3 + 1.5*iqr


# In[118]:


q1 - 1.5*iqr


# In[119]:


df.plot.scatter(x="AGE",y="SALES_PRICE")
plt.title("Age vs saleprice")
plt.show


# In[120]:


df["Transformed"] = np.square(df.AGE)
df.plot.scatter(x="Transformed",y="SALES_PRICE")


# In[121]:


df["Transformed"] = np.power(df.AGE,0.5)
df.plot.scatter(x="Transformed",y="SALES_PRICE")


# In[122]:


df["Transformed"] = np.power(df.AGE,3)
df.plot.scatter(x="Transformed",y="SALES_PRICE")


# In[123]:


df["Transformed"] = np.log(df.AGE)
df.plot.scatter(x="Transformed",y="SALES_PRICE")


# In[124]:


df = df.drop(columns=["Sale_year","Build_year","DATE_BUILD","DATE_SALE"])


# In[125]:


#since we don't see any relation between age and sales price we droping the column
df = df.drop(columns=["AGE","Transformed"])


# In[126]:


df.shape


# In[127]:


df.columns


# # 11 BUILDTYPE

# In[128]:


df.BUILDTYPE.values


# In[129]:


df.BUILDTYPE.unique()


# In[130]:


df.BUILDTYPE.value_counts()


# In[131]:


df.BUILDTYPE = df.BUILDTYPE.replace({"Other":"Others","Comercial":"Commercial"})


# In[132]:


df.BUILDTYPE.value_counts().plot.bar()


# In[133]:


df.BUILDTYPE.isnull().sum()


# In[134]:


df.groupby("BUILDTYPE")["SALES_PRICE"].mean().sort_values(ascending = True).plot.bar()


# In[135]:


#since we cannot see linear relation we going for one hot encoding for column Buildtype


# In[136]:


df = pd.get_dummies(df,columns=["BUILDTYPE"])


# In[137]:


df.columns


# # 12 UTILITY_AVAIL}

# In[138]:


df.UTILITY_AVAIL.values


# In[139]:


df.UTILITY_AVAIL.unique()


# In[140]:


df.UTILITY_AVAIL.value_counts()


# In[141]:


df.UTILITY_AVAIL=df.UTILITY_AVAIL.replace({"All Pub":"AllPub"})


# In[142]:


df.UTILITY_AVAIL.value_counts()


# In[143]:


df.UTILITY_AVAIL.value_counts().plot.bar()


# In[144]:


df.groupby("UTILITY_AVAIL")["SALES_PRICE"].mean().sort_values(ascending = True).plot.bar()


# In[145]:


utility_avail = {"ELO":0,"NoSeWa":1,"NoSewr ":2,"AllPub":3}


# In[146]:


df.UTILITY_AVAIL = df.UTILITY_AVAIL.replace(utility_avail)


# In[147]:


df.UTILITY_AVAIL.values


# In[148]:


df.columns


# In[149]:


df.UTILITY_AVAIL.value_counts()


# # 13 STREET

# In[150]:


df.STREET.values


# In[151]:


df.STREET.value_counts()


# In[152]:


df.STREET = df.STREET.replace({"NoAccess":"No Access","Pavd":"Paved"})


# In[153]:


df.STREET.value_counts()


# In[154]:


df.STREET.isnull().sum()


# In[155]:


df.groupby("STREET")["SALES_PRICE"].mean().sort_values(ascending = True).plot.bar()


# In[156]:


street = {"No Access":0,"Paved":1,"Gravel":2}
df.STREET = df.STREET.replace(street)


# In[157]:


df.columns


# # 14 	MZZONE

# In[158]:


df.	MZZONE.values


# In[159]:


df.	MZZONE.value_counts()


# In[160]:


df.	MZZONE.isnull().sum()


# In[161]:


df.groupby("MZZONE")["SALES_PRICE"].mean().sort_values(ascending = True).plot.bar()


# In[162]:


Mzzone = {"A":0,"C":1,"I":2,"RH":3,"RL":4,"RM":5}


# In[163]:


df.MZZONE = df.MZZONE.replace(Mzzone)


# In[164]:


df.	MZZONE.value_counts()


# In[165]:


df.columns


# In[166]:


#since we don't required the columns registration fee and commission fee we droping the columns


# In[167]:


df = df.drop(columns=['REG_FEE', 'COMMIS'])


# In[168]:


df.columns


# In[169]:


df.QS_ROOMS.values


# In[170]:


df.plot.scatter(x="QS_ROOMS",y="SALES_PRICE")


# In[171]:


df.plot.scatter(x="QS_BATHROOM",y="SALES_PRICE")


# In[172]:


df.plot.scatter(x="QS_BEDROOM",y="SALES_PRICE")


# In[173]:


df.plot.scatter(x="QS_OVERALL",y="SALES_PRICE")


# In[174]:


#since we don't see any relation between sale prices and QS columns we are going to drop  them


# In[175]:


df=df.drop(columns=['QS_ROOMS','QS_BATHROOM', 'QS_BEDROOM', 'QS_OVERALL'])


# In[176]:


df.columns


# In[177]:


df.shape


# In[178]:


df = df.drop_duplicates()


# In[179]:


df.shape


# In[180]:


df.N_BEDROOM = df.N_BEDROOM.astype("int64")


# In[181]:


df.N_BATHROOM = df.N_BATHROOM.astype("int64")


# In[182]:


df.info()


# # Spliting

# In[218]:


X = df[['AREA', 'INT_SQFT', 'N_BEDROOM', 'N_BATHROOM', 'N_ROOM', 'SALE_COND',
       'PARK_FACIL', 'UTILITY_AVAIL', 'STREET', 'MZZONE',
       'BUILDTYPE_Commercial', 'BUILDTYPE_House', 'BUILDTYPE_Others']]


# In[219]:


Y = df[["SALES_PRICE"]]


# In[220]:


X.shape,Y.shape


# In[221]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state = 5)


# In[222]:


X_train.shape,X_test.shape


# In[223]:


y_train.shape,y_test.shape


# In[224]:


df.head(5)


# # Scaling 

# In[225]:


from sklearn.preprocessing import StandardScaler # importing the required function
scaler = StandardScaler() #initialis
scaler.fit(X_train) # find the values of mu and sigma
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[226]:


X_train


# # Modeling

# In[192]:


from sklearn.linear_model import LinearRegression #importing all the required functions
regressor = LinearRegression() # spredicted score = m * hours + c  

regressor.fit(X_train, y_train)


# In[193]:


print(regressor.intercept_)


# In[194]:


print(regressor.coef_)


# In[195]:


y_pred = regressor.predict(X_test)


# # Evalution

# In[196]:


from sklearn import metrics 
print('R2- SCORE:', metrics.r2_score(y_test,y_pred))


# In[197]:


coeff_df = pd.DataFrame([ 1078707.50360485  , 630786.63326103 ,-1235422.79262729   ,113413.58713878,
   1736736.41233594 ,  223726.16140465,   535730.55210821  ,  71471.87715063,
    409765.91308918 ,  886342.8800555,   1349157.02489684,  -828172.34058984,
   -502877.42833827],['AREA', 'INT_SQFT', 'N_BEDROOM', 'N_BATHROOM', 'N_ROOM', 'SALE_COND','PARK_FACIL', 'UTILITY_AVAIL', 'STREET', 'MZZONE',
'BUILDTYPE_Commercial', 'BUILDTYPE_House', 'BUILDTYPE_Others'], columns=['Coefficient'])
coeff_df


# In[198]:


#Area,N_bedroom,,N_room,BUILDTYPE_Commercial are important futures.


# # Sequential Feature selection

# In[199]:


get_ipython().system('pip install scikit-learn==0.24.2')


# In[200]:


from sklearn.feature_selection import SequentialFeatureSelector
estimator = LinearRegression()
sfs = SequentialFeatureSelector(estimator, n_features_to_select=2)
sfs.fit(X_train, y_train)
print(list(zip(['AREA', 'INT_SQFT', 'N_BEDROOM', 'N_BATHROOM', 'N_ROOM', 'SALE_COND','PARK_FACIL', 'UTILITY_AVAIL', 'STREET', 'MZZONE',
'BUILDTYPE_Commercial', 'BUILDTYPE_House', 'BUILDTYPE_Others'],sfs.get_support())))


# In[201]:


X_trans = sfs.transform(X)
X_trans


# # Cross Validation

# In[202]:


from sklearn.model_selection import cross_validate
regressor = LinearRegression(normalize = True)


# In[203]:


cv_results = cross_validate(regressor, X_train, y_train, cv=10, scoring = "r2")
cv_results['test_score'].mean()


# # KNN Modeling

# In[204]:


from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import roc_auc_score
for i in [1,2,3,4,5,6,7,8,9,10,20,50]:
  knn = KNeighborsRegressor(i) #initialising the model
  knn.fit(X_train,y_train)# training the model
  y_pred = knn.predict(X_test)
  print("K value  : " , i, " score : ", 'R2- SCORE:', metrics.r2_score(y_test,y_pred))


# In[205]:


# k=4 we are getting the best R2 score


# In[227]:


from sklearn.neighbors import KNeighborsRegressor
for i in [1,2,3,4,5,6,7,8,9,10,20,50]:
    knn = KNeighborsRegressor(i)
    cv_results = cross_validate(knn, X_train, y_train, cv=10, scoring = "r2")
    print("K value  : " , i, " score : ", 'CV- SCORE:' , cv_results['test_score'].mean())


# In[207]:


#k = 5  we are best cross validation score


# In[208]:


knn = KNeighborsRegressor(5)
knn.fit(X_train,y_train)# training the model
y_pred = knn.predict(X_test)
print('R2- SCORE:', metrics.r2_score(y_test,y_pred))


# # Decession Tree

# In[209]:


from sklearn.tree import DecisionTreeRegressor


# In[210]:


from sklearn.model_selection import cross_val_score #this will help me to do cross- validation
import numpy as np

for depth in [1,2,3,4,5,6,7,8,9,10,20]:
  dt = DecisionTreeRegressor(max_depth=depth) # will tell the DT to not grow past the given threhsold
  # Fit dt to the training set
  dt.fit(X_train, y_train) # the model is trained
  trainR2score = metrics.r2_score(y_train, dt.predict(X_train)) # this is useless information - i am showing to prove a point
  dt = DecisionTreeRegressor(max_depth=depth) # a fresh model which is not trained yet
  cv_results = cross_validate(dt, X, Y, cv=10, scoring = "r2")
  
  print("Depth  : ", depth, " Training R2score : ", trainR2score, " Cross val score : " ,cv_results['test_score'].mean())


# In[ ]:





# In[211]:


# we don't prefer Decession tree algorithm because the model was leading to overfitting


# # Random Forest

# In[212]:


from sklearn.ensemble import RandomForestRegressor


# In[213]:


for depth in [1,2,3,4,5,6,7,8,9,10,20]:
    regr = RandomForestRegressor(max_depth=depth)
    regr.fit(X_train, y_train) # the model is trained
    trainR2score = metrics.r2_score(y_train, regr.predict(X_train))
    
    regr = RandomForestRegressor(max_depth=depth)
    cv_results = cross_validate(regr, X, Y, cv=10, scoring = "r2")
    print("Depth  : ", depth, " Training R2score : ", trainR2score, " Cross val score : " ,cv_results['test_score'].mean())
    


# In[214]:


#for depth = 3 We are getting 0.81 cross validation score


# # we have got best R2 score and crossvalidation score for Linear regression 0.911 and Knn with k = 5  0.95

# # using future inportance we can see two inportant features are  "Area" and "BUILDTYPE_Commercial"

# In[ ]:




