import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm

# Load the data
red_wine = pd.read_csv('data/winequality-red.csv', sep=';')
white_wine = pd.read_csv('data/winequality-white.csv', sep=';')

red_wine['type'] = 'Red'
white_wine['type'] = 'White'

variables = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
    'pH', 'sulphates', 'alcohol'
]


print_variables = [
    'Fixed Acidity', 'Volatile Acidity', 'Citric Acid', 'Residual Sugar',
    'Chlorides', 'Free Sulfur Dioxide', 'Total Sulfur Dioxide', 'Density',
    'pH', 'Sulphates', 'Alcohol'
]

save_variables = ['fa', 'va', 'ca', 'rs', 'cl', 'fsd', 'tsd', 'den', 'ph', 'sul', 'alc']

########################################
######### Dependent Variables ##########
########################################

def mean_sd_ci(dataset, feature_var_name):
    feature = dataset[feature_var_name]

    n = len(feature)

    alpha = 0.05
    z = stats.norm.ppf(1 - alpha/2)

    # Confidence Interval for the Mean
    mean_feature = np.mean(feature)
    std_feature = np.std(feature)
    se = std_feature / np.sqrt(n)

    mean_ci_lower = mean_feature - z * se
    mean_ci_upper = mean_feature + z * se

    # Confidence Interval for the Variance
    var_feature = np.var(feature)
    chi2_lower = stats.chi2.ppf(alpha / 2, df=n - 1)
    chi2_upper = stats.chi2.ppf(1 - alpha / 2, df=n - 1)

    var_ci_lower = (n - 1) * var_feature / chi2_upper
    var_ci_upper = (n - 1) * var_feature / chi2_lower

    # Confidence Interval for the Standard Deviation
    std_ci_lower = np.sqrt(var_ci_lower)
    std_ci_upper = np.sqrt(var_ci_upper)

    print(f"Mean: {mean_feature:.5f}")
    print(f"Sample Variance: {var_feature:.5f}")
    print(f"Sample Standard Deviation: {std_feature:.5f}")
    print(f"95% Confidence Interval for the Mean: ({mean_ci_lower:.5f}, {mean_ci_upper:.5f})")
    print(f"95% Confidence Interval for the Variance: ({var_ci_lower:.5f}, {var_ci_upper:.5f})")
    print(f"95% Confidence Interval for the Standard Deviation: ({std_ci_lower:.5f}, {std_ci_upper:.5f})")

def make_plots(dataset, var_arr, print_var_arr, save_variables, col):
    lowercase_color = dataset['type'][0].lower()
    for i in range(len(var_arr)):
        print(f'########## {print_var_arr[i]} ##########')
        # Histogram
        plt.figure(figsize=(8, 6))
        sns.histplot(dataset[var_arr[i]], bins=11, color=col)
        plt.title(f'Histogram of {dataset['type'][0]} Wine {print_var_arr[i]}', fontsize=16)
        plt.xlabel(f'{print_var_arr[i]}')
        plt.ylabel('Frequency')
        plt.savefig(f'images/independent/{lowercase_color}_{save_variables[i]}_hist.png', format='png', dpi=300, bbox_inches='tight')
        # plt.show()

        # QQ Plot
        plt.figure(figsize=(8, 6))
        sm.qqplot(dataset[var_arr[i]], line='s')
        plt.title(f'QQ Plot of {dataset['type'][0]} Wine {print_var_arr[i]}')
        plt.savefig(f'images/independent/{lowercase_color}_{save_variables[i]}_qq.png', format='png', dpi=300, bbox_inches='tight')
        # plt.show()

        # Box Plot
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=dataset[var_arr[i]], color=col)
        plt.title(f'Box Plot of {dataset['type'][0]} Wine {print_var_arr[i]}', fontsize=16)
        plt.xlabel(f'{print_var_arr[i]}')
        plt.savefig(f'images/independent/{lowercase_color}_{save_variables[i]}_box.png', format='png', dpi=300, bbox_inches='tight')
        # plt.show()

        # Scatter Plot
        plt.figure(figsize=(8, 6))
        sns.regplot(x=variables[i], y='quality', data=dataset, scatter_kws={'color': 'blue'}, line_kws={'color': 'red'})
        plt.title(f'{dataset['type'][0]} Wine: Plot of {print_variables[i]} and Quality', fontsize=16)
        plt.xlabel(print_variables[i])
        plt.ylabel('Quality')
        plt.savefig(f'images/independent/{lowercase_color}_{save_variables[i]}_plt.png', format='png', dpi=300, bbox_inches='tight')
        # plt.show()

        stat, p_value = stats.normaltest(dataset[var_arr[i]])

        print(f'D\'Agostino and Pearson\'s Test Statistic: {stat}')
        print(f'P-Value: {p_value}')

        alpha = 0.05
        if p_value > alpha:
            print('The distribution is likely normal (fail to reject H0).')
        else:
            print('The distribution is not normal (reject H0).')

        mean_sd_ci(dataset, var_arr[i])

        print()


make_plots(red_wine, variables, print_variables, save_variables, '#FF1100')
make_plots(white_wine, variables, print_variables, save_variables, '#FFBFBA')