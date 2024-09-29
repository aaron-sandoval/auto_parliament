import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from single_llms import names_to_abbvs


DEFAULT_FIGSIZE = (5, 3)

def plot_cdfs(
    value_counts: dict[str, pd.Series],
    show: bool = False,
) -> plt.Figure:
    """Plot CDFs for each dataset as a line plot.

    Args:
        histograms: Dictionary of Series, where each Series contains histogram data for a dataset.
    """
    n = len(value_counts)
    fig = plt.figure(figsize=DEFAULT_FIGSIZE)
    for dataset, series in value_counts.items():
        cumsum_normalized = series.values.cumsum() / series.values.sum()
        plt.plot(series.index, cumsum_normalized, label=dataset)
    plt.xticks(series.index, series.index, rotation=45, ha='right')
    plt.xlim(0, 1)  # Assuming scores are between 0 and 1
    plt.ylim(0, 1)  # CDF values are always between 0 and 1
    plt.title('Cumulative Distribution of Mean Scores')
    plt.grid(True, linestyle='--', alpha=0.7)
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
    plt.imshow(model_performance.values, cmap='YlOrRd', aspect='auto', interpolation='nearest')
    plt.colorbar(label='Score')
    
    # Set x-axis (model names) labels
    plt.xticks(range(len(model_performance.columns)), model_performance.columns, rotation=45, ha='right')
    
    # Set y-axis (dataset names) labels
    plt.yticks(range(len(model_performance.index)), model_performance.index)
    
    # Add text annotations in each cell
    for i in range(len(model_performance.index)):
        for j in range(len(model_performance.columns)):
            text_color = 'white' if model_performance.iloc[i, j] > 0.85 else 'black'
            plt.text(j, i, f'{model_performance.iloc[i, j]:.3f}',
                     ha='center', va='center', color=text_color, weight='bold')
    
    plt.title('Belief System Score by Dataset')
    plt.tight_layout()
    if show:
        plt.show()
    return fig

def plot_covariance_among_beliefs(
    covariance_by_dataset: dict[str, pd.DataFrame],
    reduce_over_datasets: bool = True,
    show: bool = False,
) -> plt.Figure:
    """Plot covariance among beliefs.

    Args:
        covariance: DataFrame with covariance data.
    """
    n = len(covariance_by_dataset)
    fig = plt.figure(figsize=DEFAULT_FIGSIZE)
    if reduce_over_datasets:
        covariance = sum(covariance_by_dataset.values()) / n
        plt.imshow(covariance, cmap='YlOrRd', aspect='auto', interpolation='nearest')
        plt.colorbar(label='Covariance')
        plt.xticks(range(len(covariance.columns)), covariance.columns, rotation=45, ha='right')
        plt.yticks(range(len(covariance.index)), covariance.index)
        # Add text annotations in each cell
        for i in range(len(covariance.index)):
            for j in range(len(covariance.columns)):
                text_color = 'white' if covariance.iloc[i, j] > 0.85 else 'black'
                plt.text(j, i, f'{covariance.iloc[i, j]:.2f}',
                         ha='center', va='center', color=text_color, weight='bold')
        plt.title('Overall Covariance among Beliefs')
    else:
        raise NotImplementedError("Not implemented yet")
        n_subplot_cols = np.floor(np.sqrt(n)).astype(int)
        n_subplot_rows = np.ceil(n/n_subplot_cols).astype(int)
        for i, (dataset, df) in enumerate(covariance_by_dataset.items()):
            plt.subplot(n_subplot_rows, n_subplot_cols, i + 1)
            plt.imshow(df, cmap='YlOrRd', aspect='auto', interpolation='nearest')
            # plt.colorbar(label='Covariance')
            plt.xticks(range(len(df.columns)), df.columns, rotation=45, ha='right')
            plt.yticks(range(len(df.index)), df.index)
        plt.title('Covariance among Beliefs by Dataset')
    plt.tight_layout()
    if show:
        plt.show()
    return fig

