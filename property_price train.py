#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[5]:


ppt=pd.read_csv(r"D:\CLASS DATASET\Property_Price_Train.csv")


# In[6]:


ppt.head()


# In[7]:


ppt.isnull().sum()[ppt.isnull().sum()>0]


# In[8]:


ppt=ppt.drop(['Id'],axis=1)
#as these columns have  high percentage of nulls drop these  columns
ppt=ppt.drop(['Lane_Type','Fence_Quality','Miscellaneous_Feature','Pool_Quality','Fireplace_Quality'],axis=1)


# In[9]:


ppt.Lot_Extent            = ppt.Lot_Extent.fillna(ppt.Lot_Extent.mean())
ppt.Brick_Veneer_Type     = ppt.Brick_Veneer_Type.fillna('None')
ppt.Brick_Veneer_Area     = ppt.Brick_Veneer_Area.fillna(ppt.Brick_Veneer_Area.mean())
ppt.Basement_Height       = ppt.Basement_Height.fillna('Gd')
ppt.Basement_Condition    = ppt.Basement_Condition.fillna('TA')
ppt.Exposure_Level        = ppt.Exposure_Level.fillna('No')
ppt.BsmtFinType1          = ppt.BsmtFinType1.fillna('Unf')
ppt.BsmtFinType2          = ppt.BsmtFinType2.fillna('Unf')
ppt.Electrical_System     = ppt.Electrical_System.fillna('SBrkr')
ppt.Garage                = ppt.Garage.fillna('Detchd')
ppt.Garage_Built_Year     = ppt.Garage_Built_Year.fillna(ppt.Garage_Built_Year.median())
ppt.Garage_Finish_Year    = ppt.Garage_Finish_Year.fillna('Unf')
ppt.Garage_Quality        = ppt.Garage_Quality.fillna('TA')
ppt.Garage_Condition      = ppt.Garage_Condition.fillna('TA')


# In[18]:


ppt[ppt.select_dtypes(include='object').columns]=ppt[ppt.select_dtypes(include='object').columns].apply(le.fit_transform)


# In[ ]:





# In[19]:


from sklearn.preprocessing import LabelEncoder #required import
le= LabelEncoder()


# In[20]:


ppt.head(3)


# In[21]:


from sklearn.model_selection import train_test_split

train_ppt,test_ppt=train_test_split(ppt,test_size=.2)


# In[22]:


train_ppt.shape


# In[23]:


test_ppt.shape


# In[24]:


train_ppt_x=train_ppt.iloc[:,0:-1]
train_ppt_y=train_ppt.iloc[:,-1]


# In[25]:


test_ppt_x=test_ppt.iloc[:,0:-1]
test_ppt_y=test_ppt.iloc[:,-1]


# In[26]:


from sklearn.linear_model import LinearRegression
linreg=LinearRegression()


# In[27]:


linreg.fit(train_ppt_x,train_ppt_y)


# In[28]:


Rsquare=linreg.score(train_ppt_x,train_ppt_y)
Rsquare


# In[29]:


N=train_ppt_x.shape[0]
K=train_ppt_x.shape[1]
Adjusted_Rsquare= 1-(1-Rsquare)*(N-1)/(N-K-1)
Adjusted_Rsquare


# In[30]:


linreg.coef_


# In[31]:


linreg.intercept_


# In[32]:


Pred_train=linreg.predict(train_ppt_x)
Pred_train


# In[34]:


Act_pre_error=pd.DataFrame()
Act_pre_error['Actual']=train_ppt_y
Act_pre_error['Predicted']=Pred_train
Act_pre_error['Error']=Act_pre_error['Actual']-Act_pre_error['Predicted']
Act_pre_error


# In[35]:


Act_pre_error.Error.mean()


# In[36]:


Act_pre_error.Error.median()


# In[37]:


Act_pre_error.Error.skew()


# In[38]:


Act_pre_error.Error.kurtosis()+3


# In[39]:


import matplotlib.pyplot as plt


# In[40]:


plt.plot(Act_pre_error.Error,'.');


# In[41]:


plt.hist(Act_pre_error.Error,bins=30,edgecolor='black');


# In[43]:


sns.regplot(x='Actual',y='Predicted',data = Act_pre_error)


# In[45]:


import numpy as np


# In[46]:


#on train
mse_train_ppt=np.mean(np.square(Act_pre_error.Error))


# In[47]:


mse_train_ppt


# In[48]:


mape_train_ppt=np.mean(np.abs(Act_pre_error.Error*100/train_ppt_y))
mape_train_ppt


# In[49]:


#on test
mse_test_ppt=np.mean(np.square(Act_pre_error.Error))
mape_test_ppt=np.mean(np.abs(Act_pre_error.Error*100/test_ppt_y))
print(mape_train_ppt)
print(mse_train_ppt)


# In[50]:


def remove_outliers (df,col,k):
    mean= df[col].mean()
    global df1
    sd= df[col].std()
    final_list= [x for x in df[col] if (x > mean - k * sd)]
    final_list= [x for x in final_list if (x < mean + k * sd)]
    df1= df.loc[df[col].isin(final_list)]; print(df1.shape)
    print('Number of outliers removed ==>', df.shape[0]- df1.shape[0])


# In[51]:


remove_outliers(ppt,'Sale_Price',2)


# In[52]:


df1.shape


# In[53]:


corr_list=[]
for cols in ppt:
    corr_list.append(ppt.Sale_Price.corr(ppt[cols]))
corr_list


# In[54]:


Feat_imp=pd.DataFrame()
Feat_imp['Features']= ppt.columns
Feat_imp['Imp']=np.abs(corr_list)
Feat_imp=Feat_imp.sort_values('Imp',ascending=False)
Feat_imp


# In[ ]:


#if there is very less correlation  between 'x' variables and 'Y' variable then there is no need to use that 'x' variable
#while building that model


# In[55]:


Feat_imp[Feat_imp.Imp>0.3].shape


# In[56]:


l1= list(Feat_imp[Feat_imp.Imp>0.3].Features)
l1

