import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

red_wine = pd.read_csv('data/winequality-red.csv', sep=';')
white_wine = pd.read_csv('data/winequality-white.csv', sep=';')

red_wine['type'] = 'red'
white_wine['type'] = 'white'

variables = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
    'pH', 'sulphates', 'alcohol'
]

def make_plots(dataset, variables, col_str, col_hex):
    scaler = MinMaxScaler()
    dataset[variables] = scaler.fit_transform(dataset[variables])

    plt.figure(figsize=(12, 8))

    sns.boxplot(data=dataset[variables], orient="v", color=col_hex)

    plt.title(f'Box Plots for Standardized {col_str} Wine Data', fontsize=16)
    plt.ylabel('Normalized Values')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(f'images/groups/{dataset['type'][0]}_box_plots.png', format='png', dpi=300, bbox_inches='tight')
    # plt.show()

make_plots(red_wine, variables, 'Red', '#FF1100')
make_plots(white_wine, variables, 'White', '#FFBFBA')