#!/usr/bin/env python
# coding: utf-8

# In[433]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:





# In[434]:


df=pd.read_csv("smartphones.csv")


# In[435]:


df.head()

df=pd.DataFrame(df)
# In[436]:


df.columns


# In[437]:


y = df['price']
print(y)


# In[ ]:





# In[ ]:





# In[ ]:




y = df['price']
# In[438]:


print(y)


# In[439]:


x = df['avg_rating']


# In[440]:


plt.scatter(x,y)
plt.show()


# In[441]:


df.isnull().sum()


# In[442]:


df['avg_rating'].fillna(5,inplace=True)


# In[443]:


df['avg_rating'].isnull().sum()


# In[444]:


df['processor_brand'].mode()


# In[445]:


k='snapdragon'


# In[446]:


df['processor_brand'].fillna(k,inplace=True)
print("ok")


# In[ ]:





# In[ ]:





# In[ ]:





# In[447]:


df['processor_brand'].isnull().sum()


# In[448]:


df['os'].mode()


# In[449]:


k='android'


# In[450]:


df['os'].fillna(k,inplace=True)


# In[451]:


df['os'].isnull().sum()


# In[452]:


k=df['num_cores'].mean()


# In[453]:


print(k)


# In[454]:


df['num_cores'].fillna(k,inplace=True)


# In[455]:


df['num_cores'].isnull().sum()


# In[456]:


k=df['processor_speed'].mean()
df['processor_speed'].fillna(k,inplace=True)
df['processor_speed'].isnull().sum()


# In[457]:


k=df['battery_capacity'].mean()
df['battery_capacity'].fillna(k,inplace=True)
df['battery_capacity'].isnull().sum()


# In[458]:


k=df['fast_charging'].mean()
df['fast_charging'].fillna(k,inplace=True)
df['fast_charging'].isnull().sum()


# In[459]:


k=df['primary_camera_front'].mean()
df['primary_camera_front'].fillna(k,inplace=True)
df['primary_camera_front'].isnull().sum()


# In[460]:


df.isnull().sum()


# In[461]:


df['os'].unique


# In[462]:


unique_os = df['os'].unique()
print(unique_os)


# In[463]:


from sklearn.preprocessing import LabelEncoder

# Create a LabelEncoder instance
label_encoder = LabelEncoder()

# Transform the 'os' column
df['os'] = label_encoder.fit_transform(df['os'])


# In[464]:


df['os']


# In[465]:


unique_os = df['os'].unique()
print(unique_os)


# In[466]:


unique_os = df['processor_brand'].unique()
print(unique_os)


# In[467]:


plt.scatter(df['price'],df['processor_speed'])
plt.show()


# In[468]:


plt.scatter(df['price'],df['battery_capacity'])
plt.show()


# In[469]:


plt.scatter(df['price'],df['ram_capacity'])
plt.show()


# In[470]:


plt.scatter(df['avg_rating'],df['ram_capacity'])


# In[471]:


df.info()


# In[472]:


df.describe()


# In[473]:


df.columns


# In[474]:


plt.boxplot(df['price'],vert=False)


# In[475]:


plt.boxplot(df['screen_size'],vert=False)


# In[476]:


numerical_columns=['price', 'avg_rating', '5G_or_not', 'num_cores', 'processor_speed', 'battery_capacity',
       'fast_charging_available', 'fast_charging', 'ram_capacity',
       'internal_memory', 'screen_size', 'refresh_rate', 'num_rear_cameras',
       'os', 'primary_camera_rear', 'primary_camera_front',
       'extended_memory_available', 'resolution_height', 'resolution_width']


# In[477]:


# Select numerical columns
  # Add all numerical columns

# Calculate Q1 (25th percentile) and Q3 (75th percentile)
Q1 = df[numerical_columns].quantile(0.25)
Q3 = df[numerical_columns].quantile(0.75)

# Calculate IQR (Interquartile Range)
IQR = Q3 - Q1

# Define a higher threshold for outliers
threshold = 5.0  # Adjust as needed

# Identify outliers based on the higher threshold
outliers = (df[numerical_columns] < (Q1 - threshold * IQR)) | (df[numerical_columns] > (Q3 + threshold * IQR))

# Remove rows containing outliers
df = df[~outliers.any(axis=1)]

# Display the updated DataFrame
print(df.describe())


# In[478]:


df.head()


# In[479]:


df.size


# In[480]:


df.shape


# In[481]:


numerical_columns=[ 'avg_rating', '5G_or_not', 'num_cores', 'processor_speed', 'battery_capacity',
       'fast_charging_available', 'fast_charging', 'ram_capacity',
       'internal_memory', 'screen_size', 'refresh_rate', 'num_rear_cameras',
       'os', 'primary_camera_rear', 'primary_camera_front',
       'extended_memory_available', 'resolution_height', 'resolution_width']


# In[482]:


from sklearn.preprocessing import MinMaxScaler

# Select numerical columns
 # Add all numerical columns

# Create a MinMaxScaler instance
scaler = MinMaxScaler(feature_range=(1, 10))  # Set the desired range

# Scale selected numerical columns
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# Display the updated DataFrame
print(df.describe())


# In[483]:


df.head()


# In[484]:


import seaborn as sns
import matplotlib.pyplot as plt

# Select only numerical columns
numerical_columns = ['price', 'avg_rating', '5G_or_not', 'processor_speed', 'battery_capacity', 
                     'internal_memory', 'screen_size', 'refresh_rate', 'num_rear_cameras',
                     'resolution_height', 'resolution_width']

# Create a correlation matrix for numerical columns
correlation_matrix = df[numerical_columns].corr()

# Plot heatmap for numerical columns
plt.figure(figsize=(12, 10))
heatmap = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap for Numerical Columns')
plt.show()


# In[485]:


correlation_with_price = df[numerical_columns].corrwith(df['price'])

# Sort the correlations in descending order
sorted_correlations = correlation_with_price.abs().sort_values(ascending=False)

# Select the top 5 highly correlated features to 'price'
top_5_correlated_features = sorted_correlations[1:6]  # Excluding 'price' itself

print("Top 5 highly correlated features to 'price':")
print(top_5_correlated_features)


# In[486]:


df1=df


# In[487]:


# Define the list of top correlated features and exclude columns
top_correlated_features = ['processor_speed', 'internal_memory', '5G_or_not', 'refresh_rate', 'avg_rating']
exclude_columns = ['brand_name', 'model', 'price']

# Create a priority dictionary for columns
priority_dict = {col: (2 if col in top_correlated_features else 1) for col in df1.columns if col not in exclude_columns}

# Calculate the weighted sum for each row based on priority
df1['total_priority'] = df1.drop(exclude_columns, axis=1).mul(df1.columns.to_series().map(priority_dict)).sum(axis=1)

# Display the DataFrame with the new 'total_priority' column
print(df1[['total_priority']])


# In[488]:


# Define the list of top correlated features and exclude columns
top_correlated_features = ['processor_speed', 'internal_memory', '5G_or_not', 'refresh_rate', 'avg_rating']
exclude_columns = ['brand_name', 'model', 'price','processor_brand']

# Create a priority dictionary for columns
priority_dict = {col: (2.0 if col in top_correlated_features else 1.0) for col in df1.columns if col not in exclude_columns}

# Calculate the weighted sum for each row based on priority
df1['total_priority'] = df1.drop(exclude_columns, axis=1).multiply(df1.columns.to_series().map(priority_dict), axis=1).sum(axis=1)

# Display the DataFrame with the new 'total_priority' column
print(df1[['total_priority']])


# In[489]:


df1.info()


# In[493]:


def top_models_below_price(df, price):
    # Filter models below the given price
    filtered_df = df[df['price'] < price]

    # Sort by total_priority and select top 5 models
    top_5_models = filtered_df.nlargest(5, 'total_priority')['model'].tolist()

    return top_5_models

# Example usage:
# Assuming 'df' is your DataFrame
price_amount =30000# Replace with the desired price

# Get top 5 models below the given price
top_models = top_models_below_price(df1, price_amount)
print(f"Top 5 models below ${price_amount}:")
print(top_models)


# In[491]:


df['price']


# In[ ]:




