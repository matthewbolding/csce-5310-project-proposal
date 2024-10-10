import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm

# Load the data
red_wine = pd.read_csv('data/winequality-red.csv', sep=';')
white_wine = pd.read_csv('data/winequality-white.csv', sep=';')

red_wine['type'] = 'red'
white_wine['type'] = 'white'

all_wine_data = pd.concat([red_wine, white_wine])

mean_values = red_wine['pH'].mean()
std_values = red_wine['pH'].std()

variables = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
    'pH', 'sulphates', 'alcohol'
]

########## Histograms ##########

fig, axes = plt.subplots(nrows=2, ncols=6, figsize=(20, 10))
axes = axes.flatten()

# Generate histograms for each variable
for i, var in enumerate(variables):
    sns.histplot(all_wine_data[var], bins=8, ax=axes[i], color='red')
    axes[i].set_title(f'{var}')
    axes[i].set_xlabel('')
    axes[i].set_ylabel('')

# Remove any extra subplot if we have fewer variables than subplots (12 in this case)
for i in range(len(variables), len(axes)):
    fig.delaxes(axes[i])

# Adjust layout to avoid overlapping
plt.tight_layout()
plt.show()

########## Normal QQ Plots ##########

fig, axes = plt.subplots(nrows=2, ncols=6, figsize=(20, 10))
axes = axes.flatten()  # Flatten to easily iterate over axes

for i, var in enumerate(variables):
    sm.qqplot(all_wine_data[var], line='s', ax=axes[i])
    axes[i].set_title(var, fontsize=14)  # Main title (dependent variable name)
    axes[i].set_xlabel('')  # Remove x-axis label
    axes[i].set_ylabel('')  # Remove y-axis label

# Remove any extra subplot if we have fewer variables than subplots (12 in this case)
for i in range(len(variables), len(axes)):
    fig.delaxes(axes[i])

# Adjust layout to avoid overlapping
plt.tight_layout()
plt.show()