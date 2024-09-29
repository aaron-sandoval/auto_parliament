import matplotlib.pyplot as plt
import pandas as pd

from single_llms import names_to_abbvs


DEFAULT_FIGSIZE = (10, 6)

def plot_cdfs(
    histograms: dict[str, pd.Series],
    show: bool = False,
) -> plt.Figure:
    """Plot CDFs for each dataset as a line plot.

    Args:
        histograms: Dictionary of Series, where each Series contains histogram data for a dataset.
    """
    n = len(histograms)
    fig = plt.figure(figsize=DEFAULT_FIGSIZE)
    for dataset, series in histograms.items():
        plt.plot(series.index, series.values.cumsum(), label=dataset)
    plt.xlabel('Mean Score')
    plt.ylabel('CDF')
    plt.legend()
    if show:
        plt.show()
    return fig


def plot_model_performance_by_dataset(
    model_performance: pd.DataFrame,
    show: bool = False,
) -> plt.Figure:
    """Plot model performance by dataset.

    Args:
        model_performance: DataFrame with model performance data.
    """
    model_performance.rename(names_to_abbvs, axis=1, inplace=True)
    fig = plt.figure(figsize=DEFAULT_FIGSIZE)
    plt.imshow(model_performance, cmap='YlOrRd', aspect='auto', interpolation='nearest')
    plt.colorbar(label='Score')
    
    # Set x-axis (model names) labels
    plt.xticks(range(len(model_performance.columns)), model_performance.columns, rotation=45, ha='right')
    
    # Set y-axis (dataset names) labels
    plt.yticks(range(len(model_performance.index)), model_performance.index)
    
    # Add text annotations in each cell
    for i in range(len(model_performance.index)):
        for j in range(len(model_performance.columns)):
            plt.text(j, i, f'{model_performance.iloc[i, j]:.2f}',
                     ha='center', va='center', color='black', weight='bold')
    
    plt.title('Belief System Score by Dataset')
    plt.tight_layout()
    if show:
        plt.show()
    return fig

def plot_covariance_among_beliefs(
    covariance: pd.DataFrame,
    show: bool = False,
) -> plt.Figure:
    """Plot covariance among beliefs.

    Args:
        covariance: DataFrame with covariance data.
    """
    fig = plt.figure(figsize=DEFAULT_FIGSIZE)
    plt.imshow(covariance, cmap='YlOrRd', aspect='auto', interpolation='nearest')
    plt.colorbar(label='Covariance')
    plt.xticks(range(len(covariance.columns)), covariance.columns, rotation=45, ha='right')
    plt.yticks(range(len(covariance.index)), covariance.index)
    plt.title('Covariance among Beliefs')
    plt.tight_layout()
    if show:
        plt.show()
    return fig

