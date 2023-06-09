# Copyright 2023 Janek Bevendorff
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import click
from extraction_benchmark.globals import *


@click.group()
def complexity():
    """
    Calculate page extraction complexities.
    """
    pass


@complexity.command()
@click.option('-d', '--dataset', type=click.Choice(['all', *DATASETS]), default=['all'], multiple=True)
def calculate(dataset):
    """
    Calculate page complexities.

    Calculate page complexities for given datasets based on the ground truth.
    """
    if 'all' in dataset:
        dataset = sorted(DATASETS)

    if not dataset:
        click.echo('No input datasets found.\n'
                   'Make sure that all datasets have been extracted correctly to a folder "datasets/raw" '
                   'under the current working directory.', err=False)
        return

    from extraction_benchmark.complexity import calculate
    calculate(dataset)


@complexity.command()
@click.option('-d', '--dataset', type=click.Choice(['all', *DATASETS]), default=['all'], multiple=True)
@click.option('-l', '--low-quantile', type=click.Choice(['0.25', '0.33', '0.5', '0.66', '0.75']),
              help='Low-complexity quantile threshold', default='0.25')
@click.option('-h', '--high-quantile', type=click.Choice(['0.25', '0.33', '0.5', '0.66', '0.75']),
              help='High-complexity quantile threshold', default='0.75')
def visualize(dataset, low_quantile, high_quantile):
    """
    Visualize the median complexity of the datasets.
    """
    if 'all' in dataset:
        dataset = sorted(DATASETS)

    from extraction_benchmark.complexity import visualize_datasets
    visualize_datasets(dataset, low_quantile, high_quantile)


@complexity.command()
@click.option('-d', '--dataset', type=click.Choice(['all', *DATASETS]), default=['all'], multiple=True)
@click.option('-p', '--parallelism', help='Number of threads to use', default=os.cpu_count())
def extract_features(dataset, parallelism):
    """
    Extract HTML features.

    Extract HTML features from ground truth pages for complexity clustering.
    """
    if 'all' in dataset:
        dataset = sorted(DATASETS)

    from extraction_benchmark.complexity import extract_page_features
    extract_page_features(dataset, parallelism)


@complexity.command()
@click.option('-d', '--dataset', type=click.Choice(['all', *DATASETS]), default=['all'], multiple=True)
@click.option('-r', '--reduce-dim', type=int,
              help='Reduce dimensionality before clustering (0 for no reduction)')
@click.option('-c', '--clusters', type=int, default=2, help='Number of clusters')
def cluster(dataset, reduce_dim, clusters):
    """
    Perform a k-means clustering.

    Perform a k-means clustering of previously extract HTML features to estimate complexity.
    """
    if 'all' in dataset:
        dataset = sorted(DATASETS)

    from extraction_benchmark.complexity import kmeans_cluster
    try:
        kmeans_cluster(dataset, reduce_dim, clusters)
    except FileNotFoundError as e:
        raise click.FileError(e.filename, 'Make sure HTML features have been calculated.')


@complexity.command()
@click.option('-q', '--quantile', type=click.Choice(['0.25', '0.33', '0.5', '0.66', '0.75']), default='0.33',
              help='Quantile boundary')
def visualize_clusters(quantile):
    """
    Visualize k-means clustering.

    Visualize k-means clustering of HTML pages and align clusters with given complexity quantile.
    """

    from extraction_benchmark.complexity import visualize_clusters
    visualize_clusters(quantile)


@complexity.command()
@click.option('-d', '--dataset', type=click.Choice(['all', *DATASETS]), default=['all'], multiple=True)
@click.option('-s', '--split-size', type=click.FloatRange(0.1, 0.9), default=0.25, help='Training split size')
@click.option('-q', '--quantile', type=click.Choice(['0.25', '0.33', '0.5', '0.66', '0.75']), default='0.5',
              help='Quantile boundary')
def classify(dataset, split_size, quantile):
    """
    Train and evaluate a logistic regression classifier.

    Train a logistic regression classifier on a split of the complexity scores and classify the remaining ones.
    """
    if 'all' in dataset:
        dataset = sorted(DATASETS)

    from extraction_benchmark.complexity import logistic_regression_classify
    try:
        logistic_regression_classify(dataset, split_size, quantile)
    except FileNotFoundError as e:
        raise click.FileError(e.filename, 'Make sure HTML features have been calculated.')


@complexity.command()
def visualize_classes():
    """
    Visualize logistic regression classification.
    """

    from extraction_benchmark.complexity import visualize_classes
    visualize_classes()
