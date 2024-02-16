# Databricks notebook source
# Databricks notebook source
import numpy as np
import pandas as pd
import matplotlib as plt

import seaborn as sns


# COMMAND ----------

df1 = spark.read.format("csv").option("header", "true").load("dbfs:/FileStore/shared_uploads/wagh.v@northeastern.edu/out-1.csv")


# COMMAND ----------

num_rows = df1.count()
print("Number of rows: ", num_rows)


# COMMAND ----------

num_columns = len(df1.columns)
print("Number of columns: ", num_columns)



# COMMAND ----------

df1.printSchema()


# COMMAND ----------


# Descriptive Statistics for Numerical Columns
df1.describe().show()

# COMMAND ----------

#Extract Year and Month from Date
from pyspark.sql.functions import year, month, col

df1 = df1.withColumn("Year", year(col("Date"))) \
         .withColumn("Month", month(col("Date")))
df1

# COMMAND ----------

#Compute the correlation between "Value" and "Transaction_count":
from pyspark.sql.functions import col
from pyspark.sql.types import IntegerType, DoubleType

# Assuming you want to convert to IntegerType, but you can also use DoubleType if fractional numbers are present or expected
df1 = df1.withColumn("Value", col("Value").cast(IntegerType())) \
         .withColumn("Transaction_count", col("Transaction_count").cast(IntegerType()))

# Retry the correlation calculation
try:
    print("Correlation coefficient:", df1.stat.corr("Value", "Transaction_count"))
except Exception as e:
    print("Error calculating correlation:", e)



# COMMAND ----------

# Since df2 is already a limited subset of df1, you might want to use it for visualization to avoid memory issues
df2_pandas = df2.toPandas()

# COMMAND ----------

# COMMAND ----------

import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'df2_pandas' is your pandas DataFrame with 'Location' and 'Transaction_count' columns

# Convert 'Transaction_count' to numeric type, handling non-numeric entries
df2_pandas['Transaction_count'] = pd.to_numeric(df2_pandas['Transaction_count'], errors='coerce')

# Drop NaN values that may have resulted from conversion
df2_pandas = df2_pandas.dropna(subset=['Transaction_count'])

# Increase the figure size for better visibility
plt.figure(figsize=(16, 8))

# Increase overall font size for the plot
sns.set(font_scale=1.2)

# Create the boxplot with locations on the x-axis and transaction counts on the y-axis
sns.boxplot(x='Location', y='Transaction_count', data=df2_pandas)

# Improve the readability of the x-axis labels with rotation
plt.xticks(rotation=90)

# Set the title of the plot
plt.title('Transaction Count Distribution by Location')

# Show the plot
plt.show()

# COMMAND ----------

#Geographical Analysis with PySpark

from pyspark.sql import SparkSession
from pyspark.sql.functions import sum as _sum, count as _count

spark = SparkSession.builder.appName("geographical_analysis").getOrCreate()

# Assuming df1 is your Spark DataFrame with the actual data

# Group by 'Location' and aggregate 'Transaction_count' and 'Value'
location_agg = df1.groupBy('Location').agg(
    _sum('Transaction_count').alias('Total_Transactions'),
    _sum('Value').alias('Total_Value'),
    _count('Transaction_count').alias('Number_of_Transactions')
).orderBy('Total_Value', ascending=False)

# Show the results
location_agg.show()


# COMMAND ----------


#Comparative Analysis by Location
# Group by both 'Location' and 'Domain' and aggregate 'Value'
location_domain_agg = df1.groupBy('Location', 'Domain').agg(
    _sum('Value').alias('Total_Value')
).orderBy('Location', 'Total_Value', ascending=False)

# Show the results
location_domain_agg.show()


# COMMAND ----------

#Domain Analysis with PySpark
# Group by 'Domain' and aggregate 'Value', then sort by the aggregated value
domain_agg = df1.groupBy('Domain').agg(
    _sum('Value').alias('Total_Value'),
    _count('Transaction_count').alias('Number_of_Transactions')
).orderBy('Total_Value', ascending=False)

# Show the results
domain_agg.show()



# COMMAND ----------

# Average transaction value every day for each domain over the year
from pyspark.sql.functions import to_date, avg

df1 = df1.withColumn('Date', to_date('Date'))

# Calculate the average transaction value per day for each domain
avg_transaction_value_per_day_domain = df1.groupBy('Date', 'Domain').agg(avg('Value').alias('AvgTransactionValue')).orderBy('Date', 'Domain')

avg_transaction_value_per_day_domain.show()


# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Sample data in a dictionary format (based on your provided output)
data = {
    'Date': ['2022-01-01', '2022-01-01', '2022-01-01', '2022-01-01', '2022-01-01',
             '2022-01-02', '2022-01-02', '2022-01-02', '2022-01-02', '2022-01-02',
             '2022-01-02', '2022-01-02', '2022-01-03', '2022-01-03', '2022-01-03',
             '2022-01-03', '2022-01-03', '2022-01-03'],
    'Domain': ['INVESTMENTS', 'MEDICAL', 'PUBLIC', 'RESTRAUNT', 'RETAIL',
               'EDUCATION', 'INTERNATIONAL', 'INVESTMENTS', 'MEDICAL', 'PUBLIC',
               'RESTRAUNT', 'RETAIL', 'EDUCATION', 'INTERNATIONAL', 'INVESTMENTS',
               'MEDICAL', 'PUBLIC', 'RESTRAUNT'],
    'AvgTransactionValue': [763030.98, 750361.55, 745098.80, 748138.16, 735011.84,
                            757351.73, 736120.70, 742187.49, 760977.26, 752134.63,
                            754181.83, 754811.89, 741130.56, 758857.82, 755376.17,
                            751057.85, 738183.03, 770697.57]
}

# Convert the dictionary to a pandas DataFrame
df = pd.DataFrame(data)

# Convert 'Date' to datetime format for proper plotting
df['Date'] = pd.to_datetime(df['Date'])

# Set the figure size and background style
plt.figure(figsize=(14, 8))
sns.set(style="whitegrid")

# Plotting
sns.lineplot(x='Date', y='AvgTransactionValue', hue='Domain', data=df, marker="o")

# Customize the plot with titles, labels, and legend
plt.title('Average Transaction Value Per Day by Domain', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Average Transaction Value', fontsize=14)
plt.legend(title='Domain', title_fontsize='13', fontsize='12', loc='upper left', bbox_to_anchor=(1, 1))
plt.xticks(rotation=45)
plt.tight_layout()

# Show the plot
plt.show()


# COMMAND ----------

#a grouped bar chart
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Sample data based on the provided output, assuming conversion to a Pandas DataFrame has been done
data = {
    'Date': ['2022-01-01', '2022-01-01', '2022-01-02', '2022-01-02', '2022-01-03', '2022-01-03'],
    'Domain': ['INVESTMENTS', 'MEDICAL', 'INVESTMENTS', 'MEDICAL', 'INVESTMENTS', 'MEDICAL'],
    'AvgTransactionValue': [763030.98, 750361.55, 742187.49, 760977.26, 755376.17, 751057.85]
}
df = pd.DataFrame(data)

# For clarity in visualization, you may choose a subset or aggregate data differently

# Creating a grouped bar chart
plt.figure(figsize=(12, 8))
sns.barplot(x='Date', y='AvgTransactionValue', hue='Domain', data=df)

# Adding chart labels and title
plt.xlabel('Date')
plt.ylabel('Average Transaction Value')
plt.title('Average Transaction Value Per Day by Domain')
plt.xticks(rotation=45)

plt.legend(title='Domain')
plt.tight_layout()
plt.show()


# COMMAND ----------

#  Average transaction value for every city/location over the year?

# Calculate the average transaction value per location over the year
avg_transaction_value_per_location = df1.groupBy('Location').agg(avg('Value').alias('AvgTransactionValue')).orderBy('Location')

avg_transaction_value_per_location.show()


# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Sample data in a dictionary format 
data = {
    'Location': ['Akola', 'Ambala', 'Amritsar', 'Ara', 'Banglore', 'Betul', 'Bhind', 'Bhopal', 'Bhuj', 
                 'Bidar', 'Bikaner', 'Bokaro', 'Bombay', 'Buxar', 'Daman', 'Delhi', 'Doda', 'Durg'],
    'AvgTransactionValue': [750687.57, 749278.56, 749028.24, 747620.41, 748846.69, 748331.26, 749056.09, 
                            751652.49, 749740.51, 749054.30, 748259.51, 748163.30, 749867.82, 750411.74, 
                            748532.25, 750385.35, 748317.29, 750731.93]
}

# Convert the dictionary to a pandas DataFrame
df = pd.DataFrame(data)

# Set the figure size and background style
plt.figure(figsize=(14, 8))
sns.set(style="whitegrid")

# Create the bar chart
sns.barplot(x='Location', y='AvgTransactionValue', data=df, palette='coolwarm')

# Customize the plot with titles, labels, and legend
plt.title('Average Transaction Value Per Location', fontsize=16)
plt.xlabel('Location', fontsize=14)
plt.ylabel('Average Transaction Value', fontsize=14)
plt.xticks(rotation=90)  # Rotate the location names for better readability

# Adjust layout for better fit and show plot
plt.tight_layout()
plt.show()


# COMMAND ----------

#violin plot 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Sample data based on your provided output, assuming conversion to a Pandas DataFrame has been done
data = {
    'Location': ['Akola', 'Ambala', 'Amritsar', 'Ara', 'Banglore', 'Betul', 'Bhind', 'Bhopal', 'Bhuj', 
                 'Bidar', 'Bikaner', 'Bokaro', 'Bombay', 'Buxar', 'Daman', 'Delhi', 'Doda', 'Durg'],
    'AvgTransactionValue': [750687.57, 749278.56, 749028.24, 747620.41, 748846.69, 748331.26, 749056.09, 
                            751652.49, 749740.51, 749054.30, 748259.51, 748163.30, 749867.82, 750411.74, 
                            748532.25, 750385.35, 748317.29, 750731.93]
}
df = pd.DataFrame(data)

# Creating the violin plot
plt.figure(figsize=(12, 10))
sns.violinplot(x='AvgTransactionValue', y='Location', data=df, scale='width', inner=None, palette='coolwarm')

# Adding chart labels and title
plt.xlabel('Average Transaction Value')
plt.ylabel('Location')
plt.title('Average Transaction Value Per Location - Violin Plot')

plt.show()


# COMMAND ----------

# If the domains could be sorted into a priority, what would be the priority list for promoting the ease of transaction for the highest active domain 
# Aggregate by Domain to calculate total transaction count and total value
domain_priority = df1.groupBy('Domain').agg(
    _sum('Transaction_count').alias('TotalTransactions'),
    _sum('Value').alias('TotalValue')
).orderBy('TotalTransactions', ascending=False)

domain_priority.show()


# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Sample data based on your provided output
data = {
    'Domain': ['PUBLIC', 'INTERNATIONAL', 'INVESTMENTS', 'EDUCATION', 'RESTRAUNT', 'MEDICAL', 'RETAIL'],
    'TotalTransactions': [212214482, 212147527, 211532374, 211454073, 211232735, 211186104, 210643016],
    'TotalValue': [107791432924, 107724396447, 107613592821, 107658704394, 107498499345, 107790980756, 107129506265]
}

# Convert the dictionary to a pandas DataFrame
df = pd.DataFrame(data)

# Create figure and axis objects with subplots()
fig, ax1 = plt.subplots(figsize=(14, 8))

# Bar plot for TotalTransactions
color = 'tab:blue'
ax1.set_xlabel('Domain', fontsize=14)
ax1.set_ylabel('Total Transactions', color=color, fontsize=14)
ax1 = sns.barplot(x='Domain', y='TotalTransactions', data=df, palette='summer', alpha=0.6)
ax1.tick_params(axis='y', labelcolor=color)
plt.xticks(rotation=45)  # Rotate the domain names for better readability

# Create a twin axis object for the TotalValue line plot
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Total Value', color=color, fontsize=14)  # we already handled the x-label with ax1
ax2 = sns.lineplot(x='Domain', y='TotalValue', data=df, sort=False, marker='o', color=color)
ax2.tick_params(axis='y', labelcolor=color)

# Further customize with titles and set layout
plt.title('Total Transactions and Total Value by Domain', fontsize=16)
fig.tight_layout()  # To adjust subplot params so the subplot(s) fits in to the figure area.

plt.show()


# COMMAND ----------

# bubble chart
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# domain_priority is your Pandas DataFrame with the converted Spark DataFrame data
data = {
    'Domain': ['PUBLIC', 'INTERNATIONAL', 'INVESTMENTS', 'EDUCATION', 'RESTRAUNT', 'MEDICAL', 'RETAIL'],
    'TotalTransactions': [212214482, 212147527, 211532374, 211454073, 211232735, 211186104, 210643016],
    'TotalValue': [107791432924, 107724396447, 107613592821, 107658704394, 107498499345, 107790980756, 107129506265]
}
df = pd.DataFrame(data)

# Normalize TotalValue for bubble size in a way that makes the sizes manageable/visible
df['BubbleSize'] = df['TotalValue'] / df['TotalValue'].max() * 1000  # Example normalization, adjust as needed

# Create a bubble chart
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='TotalTransactions', y='Domain', size='BubbleSize', legend=False, sizes=(100, 1000))

# Adding details to the chart
plt.xlabel('Total Transactions')
plt.ylabel('Domain')
plt.title('Domain Priority by Total Transactions and Value')

plt.grid(True)
plt.show()


# COMMAND ----------

# Average transaction count for each city
# Calculate the average transaction count per location
avg_transaction_count_per_location = df1.groupBy('Location').agg(avg('Transaction_count').alias('AvgTransactionCount')).orderBy('Location')

avg_transaction_count_per_location.show()


# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Sample data in a dictionary format
data = {
    'Location': ['Akola', 'Ambala', 'Amritsar', 'Ara', 'Banglore', 'Betul', 'Bhind', 'Bhopal', 'Bhuj', 
                 'Bidar', 'Bikaner', 'Bokaro', 'Bombay', 'Buxar', 'Daman', 'Delhi', 'Doda', 'Durg'],
    'AvgTransactionCount': [1469.96, 1469.83, 1474.15, 1470.29, 1475.76, 1479.90, 1478.66, 1468.42, 
                            1472.66, 1476.32, 1471.83, 1469.92, 1471.61, 1475.75, 1471.84, 1471.64, 
                            1472.42, 1469.95]
}

# Convert the dictionary to a pandas DataFrame
df = pd.DataFrame(data)

# Set the figure size and background style
plt.figure(figsize=(10, 12))
sns.set(style="whitegrid")

# Create the horizontal bar chart
sns.barplot(x='AvgTransactionCount', y='Location', data=df, palette='viridis')

# Customize the plot with titles, labels, and legend
plt.title('Average Transaction Count Per Location', fontsize=16)
plt.xlabel('Average Transaction Count', fontsize=14)
plt.ylabel('Location', fontsize=14)

# Show the plot
plt.show()


# COMMAND ----------

# Alternative graph 
#alternative visualization of the average transaction count per location, dot plot
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming df is your Pandas DataFrame with the converted Spark DataFrame data
data = {
    'Location': ['Akola', 'Ambala', 'Amritsar', 'Ara', 'Banglore', 'Betul', 'Bhind', 'Bhopal', 'Bhuj', 'Bidar', 'Bikaner', 'Bokaro', 'Bombay', 'Buxar', 'Daman', 'Delhi', 'Doda', 'Durg'],
    'AvgTransactionCount': [1469.96, 1469.83, 1474.15, 1470.29, 1475.76, 1479.90, 1478.66, 1468.42, 1472.66, 1476.32, 1471.83, 1469.92, 1471.61, 1475.75, 1471.84, 1471.64, 1472.42, 1469.95]
}
df = pd.DataFrame(data)

# Sorting the DataFrame for better visualization
df_sorted = df.sort_values('AvgTransactionCount', ascending=True)

# Creating the dot plot
plt.figure(figsize=(10, 8))
sns.stripplot(x='AvgTransactionCount', y='Location', data=df_sorted, size=10, color='dodgerblue', edgecolor='gray', linewidth=0.5)

# Adding chart labels and title
plt.xlabel('Average Transaction Count')
plt.ylabel('Location')
plt.title('Average Transaction Count Per Location - Dot Plot')

plt.tight_layout()
plt.show()

