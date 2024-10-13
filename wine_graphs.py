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

variables = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
    'pH', 'sulphates', 'alcohol'
]


##########################################
######### Independent Variables ##########
##########################################

######### Histograms ##########

# fig, axes = plt.subplots(nrows=2, ncols=6, figsize=(20, 10))
# axes = axes.flatten()

# for i, var in enumerate(variables):
#     sns.histplot(red_wine[var], bins=8, ax=axes[i], color='red')
#     axes[i].set_title(f'{var}')
#     axes[i].set_xlabel('')
#     axes[i].set_ylabel('')

# # Remove any extra subplot if we have fewer variables than subplots (12 in this case)
# for i in range(len(variables), len(axes)):
#     fig.delaxes(axes[i])

# # Adjust layout to avoid overlapping
# plt.tight_layout()
# plt.show()

# ########## Normal QQ Plots ##########

# fig, axes = plt.subplots(nrows=2, ncols=6, figsize=(20, 10))
# axes = axes.flatten()  # Flatten to easily iterate over axes

# for i, var in enumerate(variables):
#     sm.qqplot(red_wine[var], line='s', ax=axes[i])
#     axes[i].set_title(var, fontsize=14)  # Main title (dependent variable name)
#     axes[i].set_xlabel('')  # Remove x-axis label
#     axes[i].set_ylabel('')  # Remove y-axis label

# # Remove any extra subplot if we have fewer variables than subplots (12 in this case)
# for i in range(len(variables), len(axes)):
#     fig.delaxes(axes[i])

# # Adjust layout to avoid overlapping
# plt.tight_layout()
# plt.show()

########################################
######### Dependent Variables ##########
########################################

### Red Wine
plt.figure(figsize=(8, 6))
sns.histplot(red_wine['quality'], bins=11, discrete=True, color='#FF1100')
plt.xlim(0, 10)
plt.xticks(range(11))
plt.title('Histogram of Red Wine Quality', fontsize=16)
plt.xlabel('Quality')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(8, 6))
sm.qqplot(red_wine['quality'], line='s')
plt.title('QQ Plot of Red Wine Quality')
plt.show()

# Test Normality
stat, p_value = stats.normaltest(red_wine['quality'])

print(f"D'Agostino and Pearson's Test Statistic: {stat}")
print(f"P-Value: {p_value}")

alpha = 0.05
if p_value > alpha:
    print("The distribution is likely normal (fail to reject H0).")
else:
    print("The distribution is not normal (reject H0).")


# White Wine
plt.figure(figsize=(8, 6))
sns.histplot(white_wine['quality'], bins=11, discrete=True, color='#FFBFBA')
plt.xlim(0, 10)
plt.xticks(range(11))
plt.title('Histogram of White Wine Quality', fontsize=16)
plt.xlabel('Quality')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(8, 6))
sm.qqplot(white_wine['quality'], line='s')
plt.title('QQ Plot of White Wine Quality')
plt.show()

# Test Normality
stat, p_value = stats.normaltest(white_wine['quality'])

print(f"D'Agostino and Pearson's Test Statistic: {stat}")
print(f"P-Value: {p_value}")

alpha = 0.05
if p_value > alpha:
    print("The distribution is likely normal (fail to reject H0).")
else:
    print("The distribution is not normal (reject H0).")
