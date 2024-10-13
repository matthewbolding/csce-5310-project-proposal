import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

red_wine = pd.read_csv('data/winequality-red.csv', sep=';')
white_wine = pd.read_csv('data/winequality-white.csv', sep=';')

variables = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
    'pH', 'sulphates', 'alcohol'
]

def make_plots(dataset):
    scaler = MinMaxScaler()
    dataset[variables] = scaler.fit_transform(dataset[variables])

    variables.append('quality')

    sns.set(style="ticks")
    pair_plot = sns.pairplot(dataset[variables], corner=False, diag_kind='kde', plot_kws={'alpha': 0.5, 's': 20, 'edgecolor': 'k'})

    pair_plot.fig.set_size_inches(20, 15)

    # Show the plot
    plt.tight_layout()
    plt.show()

# make_plots(red_wine)
make_plots(white_wine)