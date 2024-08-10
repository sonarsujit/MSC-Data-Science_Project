#!/usr/bin/env python
# coding: utf-8

# ### Title: Impact of Employment on Mental Health Globally from Gapminder Data: Analyzing the Relationship between employment rates and suicide rates across countries?
# How do employment rates impact suicide rates globally, and does this association change when considering income per person, urbanization, and alcohol consumption?
# 
# Explotary Data Analysis.
# *We will explore the Gapminder datasets and try to gain insihts about the data using basic statistics metrics using python

# In[158]:


#importing/loading necessary libraries for data exploration

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#setting up the visual style
sns.set(style = "whitegrid")


# In[159]:


# Load dataset
# added additional column "Region" to map the countries and to be able to analyze data based on regions
df = pd.read_csv("gapminder.csv",low_memory=False)
df.head()


# In[160]:


#checking the dataset
print(f"total observations: {len(df)}")
print(f"total columns:{len(df.columns)}")
print(df.columns)


# In[161]:


#Creating own codebokk based on the Research question formulised

data = df[['region','country','incomeperperson','alcconsumption','employrate','urbanrate','suicideper100th']]
data.head()


# In[162]:


#checking the stracture of the dataset
data.info()


# In[163]:


#converting the data into correct datatype for analysis
data['incomeperperson'] = pd.to_numeric(data['incomeperperson'], errors='coerce')
data['alcconsumption'] = pd.to_numeric(data['alcconsumption'], errors='coerce')
data['employrate'] = pd.to_numeric(data['employrate'], errors='coerce')
data['urbanrate'] = pd.to_numeric(data['urbanrate'], errors='coerce')
data['suicideper100th'] = pd.to_numeric(data['suicideper100th'], errors='coerce')
data.info()


# In[164]:


# checking any missing values
# Check for missing values
missing_values = data.isnull().sum()

# Display missing values for each column
print("Missing values in each column:")
print(missing_values)


# In[165]:


data.replace(r'^\s*$', np.nan, regex=True, inplace=True)

# Optionally, check if the blanks were replaced by NaN
print(data.isnull().sum())


# In[166]:


rows_missing_data =sum([True for idx,row in data.iterrows() if any(row.isnull())])
print(f"no. of rows with missing data: {rows_missing_data}")
print(f" % of missing data: {(rows_missing_data/len(data)):.2f}")


# In[167]:


# making a copy of the data
data1 = data.copy()
data1.head()


# In[168]:


# replacing all the missing values with the group mean by region

# Function to fill NaN values with the group mean
def fill_with_group_mean(group):
    return group.fillna(group.mean())

# Apply the function to each group (grouped by 'country')
data1 = data1.groupby('region').apply(fill_with_group_mean)

# Reset index if needed
data1.reset_index(drop=True, inplace=True)

# Optionally, check if the missing values were filled
print(data1.isnull().sum())



# In[182]:


data1.head()


# ### We have processed the data and now we have a cleaned dataset with only the variables of our interetst and no missing data. We applied the replaced with group mean menthod to treat the missing data.

# ##### Renaming our data1 to a meaningful name
# 

# In[183]:


df_gm = data1.copy()
df_gm.head()


# In[184]:


# Summary statistics for all columns
df_gm.describe()


# In[185]:


#checking if the variable distribution is normal or not.
import numpy as np
import statsmodels.api as sm
import pylab

incomeperperson_1 = df_gm.incomeperperson

sm.qqplot(incomeperperson_1, line='45')
pylab.show()


suicideper100th_1 = df_gm.suicideper100th

sm.qqplot(suicideper100th_1, line='45')
pylab.show()

employrate_1 = df_gm.employrate

sm.qqplot(employrate_1, line='45')
pylab.show()


# # Consclusion
# 

# **We saw there are 213 observations and 16 columns(features) provided in the gapminder dataset as per the course material
# There is one categorical column "country" and the rest are numeric variables
# based on my reserach question: 
# Impact of Employment on Mental Health Globally from Gapminder Data: Analyzing the Relationship between employment rates and suicide rates across countries?
# Refined Research Question:
# How do employment rates impact suicide rates globally, and does this association change when considering income per person, urbanization, and alcohol consumption?
# 
# below are the list of variables that I will be considering for my research project:
#     employrate
#     suicideper100th
#     incomeperperson
#     urbanrate
#     alcconsumption
#     female employrate (additional)
#     
# #### Data Processing/Cleaning:
# On doing basic analysis and EDA we found that there were around 50 records which had one or the other missing data
# which consitituted to around 23% missing data.
# To handle missing data, we first added a region column to map the countries to respective regions
# Then using the region we used the group mean to replace the the missing data. 
# 
# #### Frequency/Distribution:
#     Looking at the frequency distribution of the three variables of interest:
#         1. incomeperperson: It is seen that majority of the countries mean incomeperperon is is upto 10000 USD 
#             with few falling between 10K to 40K and wee also see their are outliers  above 40K. Data is right skewed.
#         2. Suiciderate is showing with majority of the countries falling with a mean of 9.57 and std of 6.07 and we
#         see there are some outliers
#         3. employrate shows that in majority of the countries around 58.72 % of population age above 15+ 
#         are employed with std of 9.8. 
#         4. None of the distribution appears to be normally distributed as per teh Q-Q plot
#         
# We may have to do more data processing to see how best we can use the data to gain further insights and 
# relationships betwwen the variables

# STEP 2: Run Frequency Distributions for Selected Variables
# 
# Now that your data is cleaned, you can run frequency distributions for three selected variables: incomeperperson, employrate, and suicideper100th.

# In[186]:


# Frequency distribution for incomeperperson
income_dist = df_gm['incomeperperson'].value_counts().sort_index()
print("Frequency Distribution for Income per Person:\n", income_dist)

# Frequency distribution for employrate
employrate_dist = df_gm['employrate'].value_counts().sort_index()
print("Frequency Distribution for Employment Rate:\n", employrate_dist)

# Frequency distribution for suicideper100th
suicide_dist = df_gm['suicideper100th'].value_counts().sort_index()
print("Frequency Distribution for Suicide Rate per 100,000 people:\n", suicide_dist)


# Interpreting Your Output
# 
#     Income per Person:
#         This will show how many countries fall into different income brackets. You can observe the distribution of wealth across countries in your dataset.
# 
#     Employment Rate:
#         This distribution will provide insights into how employment rates vary globally. You may find clusters where employment rates are higher or lower.
# 
#     Suicide Rate per 100,000 People:
#         Understanding this distribution can help identify regions or countries with higher or lower suicide rates, which could be linked to factors like employment or income levels.

# # WEEk 3: Data Mangement and decisions

# In[187]:


df_gm.head()


# # Distributon Plots

# In[188]:


# Plotting the distribution for 'incomeperperson'
plt.figure(figsize=(14, 6))

plt.subplot(1, 3, 1)
sns.histplot(df_gm['incomeperperson'], kde=True, color='blue')
plt.title('Distribution of Income per Person')
plt.xlabel('Income per Person')
plt.ylabel('Frequency')

# Plotting the distribution for 'employrate'
plt.subplot(1, 3, 2)
sns.histplot(df_gm['employrate'], kde=True, color='green')
plt.title('Distribution of Employment Rate')
plt.xlabel('Employment Rate (%)')
plt.ylabel('Frequency')

# Plotting the distribution for 'suicideper100th'
plt.subplot(1, 3, 3)
sns.histplot(df_gm['suicideper100th'], kde=True, color='red')
plt.title('Distribution of Suicide Rate per 100,000')
plt.xlabel('Suicide Rate per 100,000')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


# In[189]:


# Boxplots to visualize outliers and distribution characteristics
plt.figure(figsize=(14, 6))

plt.subplot(1, 3, 1)
sns.boxplot(data=df_gm, x='incomeperperson', color='blue')
plt.title('Boxplot of Income per Person')

plt.subplot(1, 3, 2)
sns.boxplot(data=df_gm, x='employrate', color='green')
plt.title('Boxplot of Employment Rate')

plt.subplot(1, 3, 3)
sns.boxplot(data=df_gm, x='suicideper100th', color='red')
plt.title('Boxplot of Suicide Rate per 100,000')

plt.tight_layout()
plt.show()


# Summary:
# 
#     Income per Person: The distribution is highly skewed, with most countries having lower incomes and a few having very high incomes. This indicates global income inequality.
#     Employment Rate: Employment rates are more evenly distributed, suggesting that most countries have similar employment levels, with few extreme cases.
#     Suicide Rate per 100,000: The majority of countries have low suicide rates, but there are notable outliers with higher rates, potentially indicating areas where mental health support may be lacking.

# ### By binning the continuous variables into categories, we can generate more insightful visualizations that make trends and patterns easier to discern.

# In[190]:


# Ensure that columns to be binned are numeric
df_gm['incomeperperson'] = pd.to_numeric(df_gm['incomeperperson'], errors='coerce')
df_gm['employrate'] = pd.to_numeric(df_gm['employrate'], errors='coerce')
df_gm['suicideper100th'] = pd.to_numeric(df_gm['suicideper100th'], errors='coerce')
df_gm['alcconsumption'] = pd.to_numeric(df_gm['alcconsumption'], errors='coerce')

# Recoding 'incomeperperson' into bins
df_gm['income_bins'] = pd.cut(df_gm['incomeperperson'], bins=4, labels=['Low', 'Lower-Middle', 'Upper-Middle', 'High'])

# Binning 'employrate' into categories
df_gm['employrate_bins'] = pd.cut(df_gm['employrate'], bins=3, labels=['Low Employment', 'Medium Employment', 'High Employment'])

# Binning 'employrate' into categories
df_gm['alcconsumption_bins'] = pd.cut(df_gm['alcconsumption'], bins=3, labels=['Low Alcconsumption', 'Medium Alcconsumption', 'High Alcconsumption'])


# Binning 'suicideper100th' into categories
df_gm['suicide_bins'] = pd.cut(df_gm['suicideper100th'], bins=3, labels=['Low Suicide Rate', 'Medium Suicide Rate', 'High Suicide Rate'])

# Frequency distributions for the recoded variables
income_freq = df_gm['income_bins'].value_counts().sort_index()
employ_freq = df_gm['employrate_bins'].value_counts().sort_index()
acconmp_freq = df_gm['alcconsumption_bins'].value_counts().sort_index()
suicide_freq = df_gm['suicide_bins'].value_counts().sort_index()


# In[191]:


# Display the frequency distributions
print("Income per Person Frequency Distribution:\n", income_freq)
print("\nEmployment Rate Frequency Distribution:\n", employ_freq)
print("\nAlcohol Consumption Frequency Distribution:\n", acconmp_freq)
print("\nSuicide Rate per 100,000 Frequency Distribution:\n", suicide_freq)


# In[195]:


# Plotting the binned distributions
fig, axs = plt.subplots(4, 1, figsize=(12, 18))

# Count Plot for Binned Income per Person
sns.countplot(x='income_bins', data=df_gm, ax=axs[0], palette='viridis')
axs[0].set_title('Distribution of Income per Person (Binned)')
axs[0].set_xlabel('Income Category')
axs[0].set_ylabel('Number of Countries')

# Count Plot for Binned Employment Rate
sns.countplot(x='employrate_bins', data=df_gm, ax=axs[1], palette='magma')
axs[1].set_title('Distribution of Employment Rate (Binned)')
axs[1].set_xlabel('Employment Rate Category')
axs[1].set_ylabel('Number of Countries')


# Count Plot for Binned Alconsumption 
sns.countplot(x='alcconsumption_bins', data=df_gm, ax=axs[2], palette='magma')
axs[2].set_title('Distribution of Employment Rate (Binned)')
axs[2].set_xlabel('Alconsumption Category')
axs[2].set_ylabel('Number of Countries')

# Count Plot for Binned Suicide Rate per 100,000
sns.countplot(x='suicide_bins', data=df_gm, ax=axs[3], palette='coolwarm')
axs[3].set_title('Distribution of Suicide Rate per 100,000 (Binned)')
axs[3].set_xlabel('Suicide Rate Category')
axs[3].set_ylabel('Number of Countries')

plt.tight_layout()
plt.show()


# Interpretation of the Binned Distribution Plots:
# 
#     Income per Person (Binned):
#         The distribution of countries across the income categories shows a concentration in the Low Income category, with fewer countries in the Medium Income and High Income categories. This suggests that a significant number of countries in the dataset have lower average income per person, indicating a skew towards less affluent nations.
# 
#     Employment Rate (Binned):
#         The distribution of employment rates is fairly even across the three categories: Low Employment, Medium Employment, and High Employment. This indicates a diverse representation of countries with varying levels of employment, without a strong skew towards any particular category.
# 
#     Suicide Rate per 100,000 (Binned):
#         The suicide rate distribution is also quite balanced, with similar numbers of countries in each of the Low, Medium, and High Suicide Rate categories. This suggests that suicide rates vary widely across countries, with no single category overwhelmingly dominating.
#         
# These plots provide a preliminary understanding of how countries are distributed across different economic, employment, and health-related metrics. They offer insights into the diversity and spread of these variables, setting the stage for further analysis of potential relationships, such as the impact of employment rates on mental health outcomes like suicide rates. 

# # Week 4:Creating graphs for your data

# In[196]:


# Create graphs of your variables one at a time (univariate graphs).

df_gm.head()


# In[198]:


# Plotting the distribution for 'incomeperperson'
plt.figure(figsize=(14, 6))

plt.subplot(1, 4, 1)
sns.histplot(df_gm['incomeperperson'], kde=True, color='blue')
plt.title('Distribution of Income per Person')
plt.xlabel('Income per Person')
plt.ylabel('Frequency')

# Plotting the distribution for 'employrate'
plt.subplot(1, 4, 2)
sns.histplot(df_gm['employrate'], kde=True, color='green')
plt.title('Distribution of Employment Rate')
plt.xlabel('Employment Rate (%)')
plt.ylabel('Frequency')

# Plotting the distribution for 'alcconsumption'
plt.subplot(1, 4, 3)
sns.histplot(df_gm['alcconsumption'], kde=True, color='blue')
plt.title('Distribution of Alcohol consumption')
plt.xlabel('Alcohol consumption (Litres)')
plt.ylabel('Frequency')

# Plotting the distribution for 'suicideper100th'
plt.subplot(1, 4, 4)
sns.histplot(df_gm['suicideper100th'], kde=True, color='red')
plt.title('Distribution of Suicide Rate per 100,000')
plt.xlabel('Suicide Rate per 100,000')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


# In[199]:


dis1=df_gm["incomeperperson"].describe()
dis2=df_gm["employrate"].describe()
dis3=df_gm["alcconsumption"].describe()
dis4=df_gm["suicideper100th"].describe()
print(dis1)
print(dis2)
print(dis3)
print(dis3)


# In[200]:


# Plotting the distribution for 'alcconsumption'
plt.figure(figsize=(14, 6))


# Plotting the distribution for 'urbanrate'
plt.subplot(1, 3, 2)
sns.histplot(df_gm['urbanrate'], kde=True, color='green')
plt.title('Distribution of Urbanization Rate')
plt.xlabel('Urbanization Rate (%)')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

dis5=df_gm["urbanrate"].describe()
print(dis5)


# 
#     Income per Person: The distribution is highly skewed, with most countries having lower incomes and a few having very high incomes. This indicates global income inequality.
#     Employment Rate: Employment rates are more evenly distributed, suggesting that most countries have similar employment levels, with few extreme cases.
#     Suicide Rate per 100,000: The majority of countries have low suicide rates, but there are notable outliers with higher rates, potentially indicating areas where mental health support may be lacking.
#     Urbanizationrate: Most of the countries shows hig urbanization.
#     Alcohol Consumption: It appears alcohol consumption is moer or less same across countries with few countries as outliers.

# # Correlation between variables

# In[201]:


df_corr =df_gm[['incomeperperson','alcconsumption','employrate','urbanrate','suicideper100th']].corr()
df_corr


# In[202]:


plt.figure(figsize=(14, 6))

# Plotting the distribution for 'urbanrate'
plt.subplot(1, 4, 1)
sns.regplot(x = "incomeperperson" , y ="suicideper100th" , data = df_gm, fit_reg= True)

plt.subplot(1, 4, 2)
sns.regplot(x = "employrate" , y ="suicideper100th" , data = df_gm, fit_reg= True)

plt.subplot(1, 4, 3)
sns.regplot(x = "alcconsumption" , y ="suicideper100th" , data = df_gm, fit_reg= True)

plt.subplot(1, 4, 4)
sns.regplot(x = "urbanrate" , y ="suicideper100th" , data = df_gm, fit_reg= True)


plt.tight_layout()
plt.show()


# 
# ## Summary
# As per the correlation scatter plots with suicideper100th variable as the dependent variable and incomperson as independent variable, we see that there is clsuter of low income per person with high number of suicide rate. There is positive correlation between suicide rate and incomeperperson however the relationship is not significantly strong.
# 
# We see there is no relationship between the emplyment rate and the suicide rate.
# 
# However, we see there is a strong positive correlation between sucide rate and alcohol consumption and weak negative correlation between suicide and urbinization rate.

# In[ ]:




