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


# In[ ]:





# In[16]:


# Get the first column name
first_column = df.columns[0]

# Filter columns to retain only those that end with '_2012' and the first column
columns_to_retain = [first_column] + [col for col in df.columns if '_2012' in col]

# Create a new DataFrame with only the filtered columns
df_data_2012 = df[columns_to_retain]
df_data_2012.head()
print(df_data_2012.columns)
print(len(df_data_2012.columns))


# In[17]:


# Load the second CSV file (header mapping)
df_mapping = pd.read_csv('fieldmapping.csv', low_memory=False)
df_mapping.head()


# In[23]:


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


# In[24]:


# Create a dictionary for mapping coded headers to cleaned descriptions
header_mapping = pd.Series(df_mapping['Description'].values, index=df_mapping['Code']).to_dict()

# Replace the coded headers in the main DataFrame
df_data_2012.rename(columns=header_mapping, inplace=True)
df_data_2012.head()


# In[28]:


#checking the dataset
print(f"total observations: {len(df_data_2012)}")
print(f"total columns:{len(df_data_2012.columns)}")
print(df_data_2012.columns)


# In[30]:


#checking the stracture of the dataset
df_data_2012.info()


# In[39]:


data = df_data_2012.copy()
data.head(2)


# In[46]:


data.to_csv('df_data_2012.csv')


# In[35]:


# checking any missing values
# Check for missing values
data.replace(r'^\s*$', np.nan, regex=True, inplace=True)
missing_values = data.isnull().sum()

# Display missing values for each column
print("Missing values in each column:")
print(missing_values)


# In[50]:


# Updated list of columns to keep
df_filtered = data[['country',
                    ',
                    'life_expectancy_at_birth_total_years'
                    ]]

df_filtered.head()


# In[52]:


#Using LASSO regression to find the pertenent charateristics of this dataset so that
#we can use only the statistically significant variables for our analysis:
#The goal is to obtain a subset of predictor varaibles that minimizes prediction error.
# this is a shrinkage and selection method where by
# shrinkage: constraint on paratmeters that shrinks coefficients towards zero
# Selection: Identifies the most important variables associated with the response variable
data.head(2)


# In[78]:


# Importing libraries 
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LassoLarsCV


# In[92]:


data_clean = data.dropna()
data_clean.head()


# In[170]:


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

# In[173]:


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




