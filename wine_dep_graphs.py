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
