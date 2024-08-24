#!/usr/bin/env python
# coding: utf-8

# # Capstone Milestone Assignment 1: Title and Introduction to the Research Question
# ## Project Title: 
# Exploring the Impact of Healthcare Access on Life Expectancy Across Nations.
# ## Research Question: 
# How does access to healthcare services influence life expectancy in different countries?
# 

# In[2]:


#importing/loading necessary libraries for data exploration

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#setting up the visual style
sns.set(style = "whitegrid")


# In[3]:


# Load dataset
# added additional column "Region" to map the countries and to be able to analyze data based on regions
df = pd.read_csv("worldbank.csv",low_memory=False)
df.head()


# In[4]:


# Get the first column name
first_column = df.columns[0]

# Filter columns to retain only those that end with '_2012' and the first column
columns_to_retain = [first_column] + [col for col in df.columns if '_2012' in col]

# Create a new DataFrame with only the filtered columns
df_data_2012 = df[columns_to_retain]
df_data_2012.head()
print(df_data_2012.columns)
print(len(df_data_2012.columns))


# In[5]:


# Load the second CSV file (header mapping)
df_mapping = pd.read_csv('WorlbankFiledtocodemapping.csv', low_memory=False)
df_mapping.head()


# In[6]:


# Function to clean and format the description
def clean_description(description):
    # Convert to lowercase
    description = description.lower()
    # Remove brackets
    description = description.replace('(', '').replace(')', '')
    # Remove commas
    description = description.replace(',', '')
    # Replace spaces with underscores
    description = description.replace(' ', '_')
    return description

# Apply the clean_description function to the description column
df_mapping['Description'] = df_mapping['Description'].apply(clean_description)

df_mapping.head()


# In[7]:


# Create a dictionary for mapping coded headers to cleaned descriptions
header_mapping = pd.Series(df_mapping['Description'].values, index=df_mapping['Code']).to_dict()

# Replace the coded headers in the main DataFrame
df_data_2012.rename(columns=header_mapping, inplace=True)
df_data_2012.head()


# In[8]:


#checking the dataset
print(f"total observations: {len(df_data_2012)}")
print(f"total columns:{len(df_data_2012.columns)}")
print(df_data_2012.columns)


# In[9]:


#checking the stracture of the dataset
df_data_2012.info()


# In[10]:


data = df_data_2012.copy()
data.head(2)


# In[11]:


data.to_csv('df_data_2012.csv')


# In[12]:


# checking any missing values
# Check for missing values
data.replace(r'^\s*$', np.nan, regex=True, inplace=True)
missing_values = data.isnull().sum()

# Display missing values for each column
print("Missing values in each column:")
print(missing_values)


# In[13]:


#Using LASSO regression to find the pertenent charateristics of this dataset so that
#we can use only the statistically significant variables for our analysis:
#The goal is to obtain a subset of predictor varaibles that minimizes prediction error.
# this is a shrinkage and selection method where by
# shrinkage: constraint on paratmeters that shrinks coefficients towards zero
# Selection: Identifies the most important variables associated with the response variable
data.head(2)


# In[14]:


# Importing libraries 
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LassoLarsCV


# In[15]:


data_clean = data.dropna()
data_clean.head()


# In[16]:


# Table1: LASSO regression

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error
# Load your data
#data = pd.read_csv('/path/to/your/df_data_2012.csv')

# Define your feature columns and target variable
features = ['adolescent_fertility_rate_births_per_1000_women_ages_15-19',
            'birth_rate_crude_per_1000_people',
            'cause_of_death_by_communicable_diseases_and_maternal_prenatal_and_nutrition_conditions_%_of_total',
            'fertility_rate_total_births_per_woman',
            'mortality_rate_infant_per_1000_live_births',
            'mortality_rate_neonatal_per_1000_live_births',
            'mortality_rate_under-5_per_1000',
            'population_ages_0-14_%_of_total',
            'urban_population_growth_annual_%',
            'inflation_consumer_prices_annual_%',
            'rural_population_%_of_total_population',
            'access_to_electricity_%_of_population',
            'access_to_non-solid_fuel_%_of_population',
            'fixed_broadband_subscriptions_per_100_people',
            'improved_sanitation_facilities_%_of_population_with_access',
            'improved_water_source_%_of_population_with_access',
            'internet_users_per_100_people',
            'population_ages_65_and_above_%_of_total',
            'survival_to_age_65_female_%_of_cohort',
            'survival_to_age_65_male_%_of_cohort',
            'adjusted_net_national_income_per_capita_current_us$',
            'automated_teller_machines_atms_per_100000_adults',
            'gdp_per_capita_current_us$',
            'health_expenditure_per_capita_current_us$',
            'population_ages_15-64_%_of_total',
            'urban_population_%_of_total',
           'mobile_cellular_subscriptions_per_100_people']

target = 'life_expectancy_at_birth_total_years'

# Extract features and target
predvar = data_clean[features]
target = data_clean[target]
predictors=predvar.copy()

# Split data into training and test sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_test, y_train, y_test  = train_test_split(predictors, target, 
                                                              test_size=.3, random_state=123)
# standardize predictors to have mean=0 and sd=1
# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Perform Lasso regression
model = LassoCV(cv=10, precompute=False).fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Get the coefficients
lasso_coefficients = pd.DataFrame({
    'Feature': features,
    'Coefficient': model.coef_
})

print(lasso_coefficients)


# 
# ## Interpretation of Results:
# 
#     Mean Squared Error (MSE): The MSE of 1.2686 indicates how well the model's predictions match the actual values. A lower MSE implies better model performance.
#     Feature Coefficients: The Lasso regression assigns coefficients to each feature. Features with a coefficient of zero are not contributing to the model, meaning Lasso has effectively removed them as irrelevant.
# 
# #### Key Features
# 
# Based on the non-zero coefficients, the most important features in predicting life expectancy are:
# 
#     Survival to Age 65, Female (% of cohort): 4.474894
#     Survival to Age 65, Male (% of cohort): 1.784757
#     Mortality Rate, Under-5 (per 1,000 live births): -0.631172
#     Fixed Broadband Subscriptions (per 100 people): 0.534774
#     Rural Population (% of total population): -0.317197
#     Health Expenditure per Capita (current US$): 0.292277
#     Automated Teller Machines (ATMs) per 100,000 adults: 0.205724
#     Fertility Rate, Total (births per woman): -0.067180
# 
# #### Feature Selection
# 
# considering the following features for further analysis:
# 
#     Survival to Age 65 (Female and Male): Strongly positive correlation with life expectancy, indicating a direct relationship.
#     Mortality Rate, Under-5: Strong negative correlation, which is expected as higher mortality rates generally lead to lower life expectancy.
#     Fixed Broadband Subscriptions and ATMs: Positive association with life expectancy, potentially indicating the influence of infrastructure and economic factors.
#     Rural Population: Negative association, suggesting that countries with higher rural populations might have lower life expectancy due to limited access to healthcare.
#     Health Expenditure per Capita: Positive correlation, which aligns with the idea that higher health expenditure leads to better health outcomes.
# 
# Additionaly below features asper Lasso regression is no important, however given access to these features does have some impact on the life expectancy, I would like to iclude these features as well for furhter analysis.
# 
# * Access to Electricity, Non-Solid Fuel, Improved Sanitation, and Water 
# 
# Features to Drop
# 
#     The features with a coefficient of zero have been deemed irrelevant by the Lasso model for predicting life expectancy and should likely be removed from further analysis. These include:
#         Adolescent Fertility Rate
#         Birth Rate
#         Internet Users per 100 People
#         GDP per Capita
#         Urban Population (both growth and total %)

# In[17]:


#Table 1: Pearson Correlation values and relative p values

import scipy.stats as stats
features = ['life_expectancy_at_birth_total_years',
            'mortality_rate_under-5_per_1000',
            'rural_population_%_of_total_population',
            'fixed_broadband_subscriptions_per_100_people',
            'improved_sanitation_facilities_%_of_population_with_access',
            'improved_water_source_%_of_population_with_access',
            'survival_to_age_65_female_%_of_cohort',
            'gdp_per_capita_current_us$',
            'health_expenditure_per_capita_current_us$']

correlations = pd.DataFrame(index=features, columns=['life_expectancy_at_birth_total_years'])
p_values = pd.DataFrame(index=features, columns=['life_expectancy_at_birth_total_years'])

# Calculate Pearson correlation and p-values
for feature in features:
    corr, p_val = stats.pearsonr(data_clean['life_expectancy_at_birth_total_years'], data_clean[feature])
    correlations.loc[feature] = corr
    p_values.loc[feature] = p_val

# Display the correlation and p-values
results = pd.concat([correlations, p_values], axis=1)
results.columns = ['Pearson Correlation', 'p-value']

# Display the results
results


# Week 3:
# # Capstone Milestone Assignment 2: Preliminary Results
# Submit a blog entry that includes 1) a description of your preliminary statistical analyses and 2) some plots or graphs to help you convey the message.
# 
# Based on the above analysis, I have decided to focus on the below list of variables to try and answer my research question. The primary feature that i will consider is the heath expenditure to predict the life expectancy and other features as secondary variables.
# 
# Features:
# * life_expectancy_at_birth_total_years
# * Survival to Age 65, Female (% of cohort)
# * Survival to Age 65, Male (% of cohort)
# * Mortality Rate, Under-5 (per 1,000 live births)
# * Fixed Broadband Subscriptions (per 100 people)
# * Rural Population (% of total population)
# * Health Expenditure per Capita (current US$)

# In[18]:


# Filtering to create a subset of data with just the required featuures for my research

df_filtered = data[['country','survival_to_age_65_female_%_of_cohort',
                        'survival_to_age_65_male_%_of_cohort',
                        'mortality_rate_under-5_per_1000',
                        'fixed_broadband_subscriptions_per_100_people',
                        'rural_population_%_of_total_population',
                        'health_expenditure_per_capita_current_us$',
                        'life_expectancy_at_birth_total_years']]

df_filtered.reset_index(inplace = True, drop = True)


# In[19]:


df_filtered.head()


# In[20]:


print(len(df_filtered))
print(len(df_filtered.columns))


# In[21]:


# Check for missing values
df_filtered.replace(r'^\s*$', np.nan, regex=True, inplace=True)
missing_values = df_filtered.isnull().sum()

# Display missing values for each column
print("Missing values in each column:")
print(missing_values)


# In[22]:


# dropping records with missing values and considerring the ste as my sample
df_sample = df_filtered.dropna()
len(df_sample)


# In[23]:


# Descriptive statistics
df_sample.describe().T


# Descriptive Statistics:
# 
# Table 1 shows descriptive statistics for life_expectancy globbaly and the quantitative predictors. The average life_expectancy globaly are 71.12 (sd=8.22), with a minimum of 48.85 users and a maximum of 83.09.

# In[24]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error
# Load your data
#data = pd.read_csv('/path/to/your/df_data_2012.csv')

# Define your feature columns and target variable
features = ['mortality_rate_under-5_per_1000',
            'rural_population_%_of_total_population',
            'fixed_broadband_subscriptions_per_100_people',
            'survival_to_age_65_female_%_of_cohort',
            'survival_to_age_65_male_%_of_cohort',
            'health_expenditure_per_capita_current_us$',
            ]

target = 'life_expectancy_at_birth_total_years'

# Extract features and target
predvar = df_sample[features]
target = df_sample[target]
predictors=predvar.copy()

# Split data into training and test sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_test, y_train, y_test  = train_test_split(predictors, target, 
                                                              test_size=.3, random_state=123)
# standardize predictors to have mean=0 and sd=1
# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Perform Lasso regression
model = LassoCV(cv=10, precompute=False).fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Get the coefficients
lasso_coefficients = pd.DataFrame({
    'Feature': features,
    'Coefficient': model.coef_
})

print(lasso_coefficients)


# In[52]:


# MSE from training and test data
from sklearn.metrics import mean_squared_error
train_error = mean_squared_error(y_train, model.predict(X_train_scaled))
test_error = mean_squared_error(y_test, model.predict(X_test_scaled))
print ('training data MSE')
print(train_error)
print ('test data MSE')
print(test_error)


# R-square from training and test data
rsquared_train=model.score(X_train_scaled,y_train)
rsquared_test=model.score(X_test_scaled,y_test)
print ('training data R-square')
print(rsquared_train)
print ('test data R-square')
print(rsquared_test)


# In[26]:





# MSE: selected model is more accurate in life expectancy prediction using the test data compared to training dataset
# 
# Rswaured value of 0.99 and 0.98 indicated that the selected model explained 99% of the variance for the life expentancy in both the training and test datasets.

# In[50]:


#plot coefficients progression

from sklearn.linear_model import Lasso
# Get the range of alphas from the LassoCV model
alphas = model.alphas_

# Initialize an empty list to store coefficients
coefs = []

# Fit a Lasso model for each alpha and store the coefficients
for alpha in alphas:
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_train_scaled, y_train)
    coefs.append(lasso.coef_)

# Convert the coefficients list to a numpy array for plotting
coefs = np.array(coefs)

# Plot the coefficient progression
plt.figure(figsize=(8, 4))
m_log_alphas = -np.log10(alphas)
plt.plot(m_log_alphas, coefs)
plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k', label='alpha CV')
plt.ylabel('Regression Coefficients')
plt.xlabel('-log(alpha)')
plt.title('Regression Coefficients Progression for Lasso Paths')
plt.legend()
plt.show()


# In[51]:


# plot mean square error for each fold
# Get the mean squared errors for each fold and each alpha
mse_path = model.mse_path_

# Get the range of alphas from the LassoCV model
alphas = model.alphas_

# Plotting the mean squared error for each fold
plt.figure(figsize=(8, 4))
for i in range(mse_path.shape[1]):  # iterate over the number of folds
    plt.plot(-np.log10(alphas), mse_path[:, i], label=f'Fold {i+1}')

plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k', label='alpha CV')
plt.xlabel('-log(alpha)')
plt.ylabel('Mean Squared Error')
plt.title('Mean Squared Error for each fold during LassoCV')
plt.legend()
plt.show()


# We can see that there is variablity across the individual cross valdiation folds in the training datasetbut the change in the mean suared error as variables are added to the model follows the same pattern for each fold.
# Inititially it decreses rapidly and then levels off to a point at which adding more predictors doesn't lead to much reduction in the mean square error

# In[29]:


#Feature Coefficient Plot:
import matplotlib.pyplot as plt
import seaborn as sns

sns.barplot(x='Coefficient', y='Feature', data=lasso_coefficients[lasso_coefficients['Coefficient'] != 0])
plt.title('Lasso Regression Coefficients')
plt.show()


# In[30]:


#Correlation heatmap
df_sample_corr = df_sample.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(df_sample_corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()


# # Bivariate Analysis:

# In[40]:


#Table 1: Pearson Correlation values and relative p values

import scipy.stats as stats
features = ['life_expectancy_at_birth_total_years',
            'mortality_rate_under-5_per_1000',
            'rural_population_%_of_total_population',
            'fixed_broadband_subscriptions_per_100_people',
            'survival_to_age_65_female_%_of_cohort',
            'survival_to_age_65_male_%_of_cohort',
            'health_expenditure_per_capita_current_us$']

correlations = pd.DataFrame(index=features, columns=['life_expectancy_at_birth_total_years'])
p_values = pd.DataFrame(index=features, columns=['life_expectancy_at_birth_total_years'])

# Calculate Pearson correlation and p-values
for feature in features:
    corr, p_val = stats.pearsonr(df_sample['life_expectancy_at_birth_total_years'], df_sample[feature])
    correlations.loc[feature] = corr
    p_values.loc[feature] = p_val

# Display the correlation and p-values
results = pd.concat([correlations, p_values], axis=1)
results.columns = ['Pearson Correlation', 'p-value']

# Display the results
results


# In[47]:


# correlation and regression plot
plt.figure(figsize=(14, 8))

plt.subplot(2, 3, 1)
sns.regplot(x = "mortality_rate_under-5_per_1000" , y ="life_expectancy_at_birth_total_years" ,
            data = df_sample, fit_reg= True)
plt.title('Life expectancy vs Mortality rate under 5 per 1000')

plt.subplot(2, 3, 2)
sns.regplot(x = "rural_population_%_of_total_population" , y ="life_expectancy_at_birth_total_years" ,
            data = df_sample, fit_reg= True)
plt.title('Life expectancy vs rural_population_%_of_total_population')

plt.subplot(2, 3, 3)
sns.regplot(x = "fixed_broadband_subscriptions_per_100_people" , y ="life_expectancy_at_birth_total_years" ,
            data = df_sample, fit_reg= True)
plt.title('Life expectancy vs Fixed_broadband_subscriptions_per_100_people')

plt.subplot(2, 3, 4)
sns.regplot(x = "survival_to_age_65_female_%_of_cohort" , y ="life_expectancy_at_birth_total_years" ,
            data = df_sample, fit_reg= True)
plt.title('Life expectancy vs survival_to_age_65_female')

plt.subplot(2, 3, 5)
sns.regplot(x = "survival_to_age_65_male_%_of_cohort" , y ="life_expectancy_at_birth_total_years" ,
            data = df_sample, fit_reg= True)
plt.title('Life expectancy vs survival_to_age_65_male')

plt.subplot(2, 3, 6)
sns.regplot(x = "health_expenditure_per_capita_current_us$" , y ="life_expectancy_at_birth_total_years" ,
            data = df_sample, fit_reg= True)
plt.title('Life expectancy vs health_expenditure_per_capita')

plt.tight_layout()
plt.show()


# In[36]:


len(data.columns)


# In[37]:


df_sample.to_csv('df_sample.csv')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




