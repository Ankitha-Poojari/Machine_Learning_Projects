#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import warnings

from sklearn.preprocessing import StandardScaler,OneHotEncoder,OrdinalEncoder

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, precision_score, recall_score

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier


# In[2]:


# to disable warnings
warnings.filterwarnings('ignore')


# In[4]:


data = pd.read_csv('HR_Dataset.csv')


# In[5]:


data.sample(2)


# left is the target variable, i.e where the employee has left the company or not. Since the values are 0 and 1 it is a binary classification problem

# In[6]:


data.columns


# 'Departments ' removing the space

# In[7]:


data.rename(columns={'Departments ':'departments'}, inplace=True)


# In[8]:


data.columns


# In[9]:


data.shape
# (rows,colums)


# In[10]:


data.info()


# In[11]:


data.isnull().sum()


# there are no null values in the entire dataset

# In[12]:


# describe method is used to find descriptive statistics of the dataset
data.describe()


# # Taking care of duplicate values

# In[13]:


data.duplicated().any()


# In[14]:


# displaying the duplicated rows
data[data.duplicated()]


# In[15]:


# dropping the duplicated rows
data = data.drop_duplicates()


# In[16]:


data.shape


# In[17]:


data.duplicated().any()


# In[18]:


14999-11991


# we have sucessfully dropped 3008 duplicate cells

# In[20]:


data['left'].value_counts().plot(kind='bar')


# by this we can determine that our dataset is imbalanced since one class has high number of observations and other class has low number of observation

# # Storing feature matrix in X and response or target in vector y

# In[21]:


X = data.drop(columns=['left'])
y = data['left']


# In[33]:


X


# In[22]:


y


# # Column Transformer and Pipeline

# In[23]:


data.head(1)


# we observe that the scale used for number_project and average_monthly are different so standardization ensures that all features are equally important during model training

# In[24]:


# we are going to use standardscalar for all the numerical values available in the data set
# standardizing the numerical data using standardscalar
preprocessor = ColumnTransformer(transformers=[
    ('num',StandardScaler(),['satisfaction_level',
                            'last_evaluation',
                            'number_project',
                            'average_montly_hours',
                            'time_spend_company',
                            'Work_accident', 'promotion_last_5years']),
    ('nominal',OneHotEncoder(),['departments']),
    ('ordinal',OrdinalEncoder(),['salary'])
    
],remainder='passthrough')


# nominal features are those columns which have no inherent order or ranking like departments.We want to prevent the model from thinking one department is greater then other.
# Ordinal feature have inherent order i.e for example low, medium and high. OrdinalEncoder converts these categories to numerical values while preserving the order. By default other columns which are not mentioned in the column transformer will be dropped but we don't want to drop it here

# # Building a machine learning pipeline
# pipeline is a series of interconnected processing steps where the output of one step serves as the input to the next step. The main purpose of doing this is to streamline the entire process of execution

# In[25]:


# our preprocessor is the columntransformer
pipeline = Pipeline([('preprocessor',preprocessor),
         ('model',LogisticRegression())
         ])


# we have sucessfully created the pipeline where the first step is preprocessor and the second step is model, i.e output of the first step will act as the input to the second step

# # Visualizing the pipeline

# In[26]:


from sklearn import set_config


# In[27]:


set_config(display='diagram')


# In[28]:


pipeline


# # Splitting the dataset into Training set and Testing set
# here 20% of the dataset is used for testing and 80% is used for training also since out dataset is imbalanced dataset we need to use a parameter stratify which will ensure the class distribution in the original dataset is preserved in both training as well as testing subsets

# In[29]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20, random_state=42,stratify=y)


# In[32]:


# training the created pipeline on X_train and y_train
pipeline.fit(X_train,y_train)


# # Performing predictions on the created pipeline using unseen samples

# In[33]:


y_pred = pipeline.predict(X_test)


# In[34]:


# accaccuracy_score(actual values, predicted values)
accuracy_score(y_test,y_pred)


# Logistic regression is around 0.8370154230929554 accurate 

# since our dataset is imbalanced we can't just rely on accuracy we need to check the precision as well

# In[35]:


precision_score(y_test,y_pred)


# the logistic regression model used is 52% precised

# In[36]:


recall_score(y_test,y_pred)


# compared to the accuracy score the precision as well as the recall scores are very less this is because our dataset is imbalanced

# # using other machine learning models in a single step
# 

# In[56]:


# def model_scorer(model_name, model instances)
def model_scorer(model_name,model):
    
    # creating an empty python list to store the accuracy, precision and recall score
    
    output=[]
    
    # we going to append the model name to the empty list
    
    output.append(model_name)
    
    pipeline = Pipeline([
        ('preprocessor',preprocessor),
         ('model',model)])
    
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20, random_state=42,stratify=y)
    
    pipeline.fit(X_train,y_train)
    
    y_pred = pipeline.predict(X_test)
    
    output.append(accuracy_score(y_test,y_pred))
    
    output.append(precision_score(y_test,y_pred))
    
    output.append(recall_score(y_test,y_pred))
    
    return output



# # Creating a python dictionary with different models in it

# In[57]:


model_dict={
    'log':LogisticRegression(),
    'decision_tree':DecisionTreeClassifier(),
    'random_forest':RandomForestClassifier(),
    'XGB':XGBClassifier()
}


# In[58]:


model_output=[]
for model_name, model in model_dict.items():
    model_output.append(model_scorer(model_name, model))


# In[60]:


model_output


# from the output we can determine that random forest is giving us the best result

# # Training the pipeline with random forest on the entire dataset

# In[61]:


preprocessor = ColumnTransformer(transformers=[
    ('num',StandardScaler(),['satisfaction_level',
                            'last_evaluation',
                            'number_project',
                            'average_montly_hours',
                            'time_spend_company',
                            'Work_accident', 'promotion_last_5years']),
    ('nominal',OneHotEncoder(),['departments']),
    ('ordinal',OrdinalEncoder(),['salary'])
    
],remainder='passthrough')


# In[62]:


pipeline = Pipeline([('preprocessor',preprocessor),
         ('model',RandomForestClassifier())
         ])


# In[63]:


# training the pipeline with entire dataset
pipeline.fit(X,y)


# We have successfully trained our pipeline with the best model that is randomforestclassifier

# In[68]:


# dataframe used to predict
sample = pd.DataFrame({
    'satisfaction_level':0.38, 
    'last_evaluation':0.53, 
    'number_project':2,
    'average_montly_hours':157, 
    'time_spend_company':3,
    'Work_accident':0, 
    'promotion_last_5years':0, 
    'departments':'sales', 
    'salary':'low'
},index=[0])


# In[70]:


result = pipeline.predict(sample)

if result ==1:
    print("This particular employee may leave the organization ")
else:
    print("This particular employee may stay with the organization")


# # Save the model 

# In[71]:


import pickle


# In[73]:


with open('pipeline.pk1','wb') as f:
    pickle.dump(pipeline,f)


# In[74]:


with open('pipeline.pk1','rb') as f:
    pipeline_saved = pickle.load(f)


# In[75]:


result = pipeline_saved.predict(sample)

if result ==1:
    print("This particular employee may leave the organization ")
else:
    print("This particular employee may stay with the organization")


# In[ ]:




