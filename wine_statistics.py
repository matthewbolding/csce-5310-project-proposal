import pandas as pd
import numpy as np
from scipy import stats

# Load the data
red_wine = pd.read_csv('data/winequality-red.csv', sep=';')
white_wine = pd.read_csv('data/winequality-white.csv', sep=';')

red_quality_mean_values = red_wine['quality'].mean()
red_quality_std_values = red_wine['quality'].std()
red_quality_var_values = np.pow(red_quality_std_values, 2)

print(f'red_quality_mean_values: {red_quality_mean_values:.3f}')
print(f'red_quality_std_values: {red_quality_std_values:.3f}')
print(f'red_quality_var_values: {red_quality_var_values:.3f}')

white_quality_mean_values = white_wine['quality'].mean()
white_quality_std_values = white_wine['quality'].std()
white_quality_var_values = np.pow(white_quality_std_values, 2)

print(f'white_quality_mean_values: {white_quality_mean_values:.3f}')
print(f'white_quality_std_values: {white_quality_std_values:.3f}')
print(f'white_quality_var_values: {white_quality_var_values:.3f}')

quality_t_value = (red_quality_mean_values - white_quality_mean_values) / \
                    np.sqrt((np.pow(red_quality_std_values, 2) / len(red_wine)) + (np.pow(white_quality_std_values, 2) / len(white_wine)))
print(f't-value: {quality_t_value:.3f}')


def mean_sd_ci(dataset):
    # Extract the 'quality' variable
    quality = dataset['quality']

    # Sample size
    n = len(quality)

    # Confidence level (95%)
    alpha = 0.05
    z = stats.norm.ppf(1 - alpha/2)  # Z-score for 95% confidence level

    # 1. Confidence Interval for the Mean
    mean_quality = np.mean(quality)
    std_quality = np.std(quality, ddof=1)  # Sample standard deviation
    se = std_quality / np.sqrt(n)  # Standard error

    # Compute the confidence interval for the mean
    mean_ci_lower = mean_quality - z * se
    mean_ci_upper = mean_quality + z * se

    # 2. Confidence Interval for the Variance
    var_quality = np.var(quality, ddof=1)  # Sample variance
    chi2_lower = stats.chi2.ppf(alpha / 2, df=n - 1)
    chi2_upper = stats.chi2.ppf(1 - alpha / 2, df=n - 1)

    # Compute the confidence interval for the variance
    var_ci_lower = (n - 1) * var_quality / chi2_upper
    var_ci_upper = (n - 1) * var_quality / chi2_lower

    # 3. Confidence Interval for the Standard Deviation
    std_ci_lower = np.sqrt(var_ci_lower)
    std_ci_upper = np.sqrt(var_ci_upper)

    # Print the results
    print(f"95% Confidence Interval for the Mean: ({mean_ci_lower:.3f}, {mean_ci_upper:.3f})")
    print(f"95% Confidence Interval for the Variance: ({var_ci_lower:.3f}, {var_ci_upper:.3f})")
    print(f"95% Confidence Interval for the Standard Deviation: ({std_ci_lower:.3f}, {std_ci_upper:.3f})")


print("Red Wine...")
mean_sd_ci(red_wine)

print("White Wine...")
mean_sd_ci(white_wine)
