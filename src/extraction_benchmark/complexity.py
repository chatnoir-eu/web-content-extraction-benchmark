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

import os.path
from collections import defaultdict
from multiprocessing import Pool
import re

import click
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from resiliparse.parse.html import HTMLTree

from extraction_benchmark.dataset_readers import read_datasets
from extraction_benchmark.globals import *
from extraction_benchmark import plt
from extraction_benchmark.util import tokenize_words


def calculate(datasets):
    """
    Calculate page complexities for pages in the given datasets based on the ground truth.

    :param datasets: list of dataset names
    """
    complexity_total = pd.DataFrame(columns=['complexity'])
    complexity_total.index.name = 'hash_key'
    quantile_labels = [0.25, 0.33, 0.5, 0.66, 0.75]

    os.makedirs(METRICS_COMPLEXITY_PATH, exist_ok=True)

    with click.progressbar(datasets, label='Calculating page complexity scores') as ds_progress:
        for ds in ds_progress:
            tokens_truth = {}
            tokens_src = {}
            for h, truth in read_datasets([ds], True):
                tokens_truth[h] = len(tokenize_words(truth['plaintext']))
            for h, src in read_datasets([ds], False):
                if h not in tokens_truth:
                    continue
                # Extract all text tokens except script / style
                tree = HTMLTree.parse(src['html'])
                for e in tree.body.query_selector_all('script, style'):
                    e.decompose()
                tokens_src[h] = len(tokenize_words(tree.body.text))

            tokens_truth = pd.DataFrame.from_dict(tokens_truth, orient='index')
            tokens_src = pd.DataFrame.from_dict(tokens_src, orient='index')

            out_path_ds = os.path.join(METRICS_COMPLEXITY_PATH, ds)
            os.makedirs(out_path_ds, exist_ok=True)

            complexity = 1 - (tokens_truth / tokens_src).clip(lower=0, upper=1)
            complexity.index.name = 'hash_key'
            complexity.columns = ['complexity']
            complexity.to_csv(os.path.join(out_path_ds, f'{ds}_complexity.csv'))
            complexity['dataset'] = ds
            quantiles = complexity['complexity'].quantile(quantile_labels)
            quantiles.to_csv(os.path.join(out_path_ds, f'{ds}_complexity_quantiles.csv'))

            complexity_total = pd.concat([complexity_total, complexity])

    complexity_total.reset_index(inplace=True)
    complexity_total.set_index(['hash_key', 'dataset'], inplace=True)
    quantiles = complexity_total.quantile(quantile_labels)
    quantiles.to_csv(os.path.join(METRICS_COMPLEXITY_PATH, f'complexity_quantiles.csv'))
    complexity_total.to_csv(os.path.join(METRICS_COMPLEXITY_PATH, f'complexity.csv'))

    click.echo(f'Complexity scores written to "{METRICS_COMPLEXITY_PATH}".')


_WS_RE = re.compile(r'\s+', flags=re.UNICODE | re.MULTILINE)


def extract_html_features(html):
    tree = HTMLTree.parse(html)
    for e in tree.body.query_selector_all('script, style, noscript'):
        e.decompose()
    text = _WS_RE.sub(' ', tree.body.text)

    features = defaultdict(float)

    all_tags = tree.body.query_selector_all('*')

    n_tags = len(all_tags)
    if n_tags != 0:
        features['h1'] = len(tree.body.query_selector_all('h1')) / n_tags
        features['h2'] = len(tree.body.query_selector_all('h2')) / n_tags
        features['h3'] = len(tree.body.query_selector_all('h3')) / n_tags
        features['h4'] = len(tree.body.query_selector_all('h4')) / n_tags
        features['h5'] = len(tree.body.query_selector_all('h5')) / n_tags
        features['h6'] = len(tree.body.query_selector_all('h6')) / n_tags
        features['p'] = len(tree.body.query_selector_all('p')) / n_tags
        features['ul'] = len(tree.body.query_selector_all('li')) / n_tags
        features['table'] = len(tree.body.query_selector_all('table')) / n_tags
        features['a'] = len(tree.body.query_selector_all('a')) / n_tags
        features['div'] = len(tree.body.query_selector_all('div')) / n_tags
        features['br'] = len(tree.body.query_selector_all('br')) / n_tags
        features['strong'] = len(tree.body.query_selector_all('strong')) / n_tags
        features['em'] = len(tree.body.query_selector_all('em')) / n_tags

    features['html_to_non_html'] = n_tags / len(tokenize_words(text))

    return features


def calculate_dataset_features(dataset):
    df = pd.DataFrame()
    for hash_key, data in read_datasets([dataset], False):
        features = extract_html_features(data['html'])
        s = pd.Series(features, name=hash_key)
        df = pd.concat([df, s.to_frame().T])
    df.index.name = 'hash_key'
    out_dir = os.path.join(HTML_FEATURES_PATH, dataset)
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, f'{dataset}_html_features.csv'))


def tsne_reduce_dim(X, n_components):
    return TSNE(n_components=n_components,
                learning_rate='auto',
                init='random',
                perplexity=30,
                method='barnes_hut',
                n_jobs=-1,
                verbose=1).fit_transform(X)


def pca_reduce_dim(X, n_components):
    return PCA(n_components=n_components).fit_transform(X)


def get_kmeans_labels(X, n_clusters):
    # noinspection PyProtectedMember,PyUnresolvedReferences
    return KMeans(n_clusters=n_clusters, max_iter=500, n_init=30).fit(X).labels_


def extract_page_features(dataset, parallelism):
    with Pool(processes=parallelism) as pool:
        with click.progressbar(pool.imap_unordered(calculate_dataset_features, dataset),
                               length=len(dataset), label='Extracting dataset features') as progress:
            for _ in progress:
                pass


# noinspection DuplicatedCode
def kmeans_cluster(dataset, reduce_dim, n_clusters):
    df_features = pd.DataFrame()
    df_complexity = pd.DataFrame()
    with click.progressbar(dataset, label='Loading datasets') as progress:
        for ds in progress:
            df_tmp = pd.read_csv(os.path.join(HTML_FEATURES_PATH, ds, f'{ds}_html_features.csv'))
            df_tmp['dataset'] = ds
            df_features = pd.concat([df_features, df_tmp], ignore_index=True)

            df_tmp = pd.read_csv(os.path.join(METRICS_COMPLEXITY_PATH, ds, f'{ds}_complexity.csv'))
            df_tmp['dataset'] = ds
            df_complexity = pd.concat([df_complexity, df_tmp], ignore_index=True)

    df_features.set_index(['hash_key', 'dataset'], inplace=True)
    df_complexity.set_index(['hash_key', 'dataset'], inplace=True)

    df_features = df_features.join(df_complexity, how='inner')

    scaler = StandardScaler()
    X = scaler.fit_transform(df_features.drop(columns='complexity'))

    if reduce_dim:
        X = pca_reduce_dim(X, reduce_dim)

    click.echo('Clustering datapoints...')
    labels = get_kmeans_labels(X, n_clusters)
    # Ensure cluster labels are aligned with quantiles
    if sum(labels[labels == 1]) < len(labels[labels == 0]):
        labels = 1 - labels
    df_features['kmeans_label'] = labels
    df_features.to_csv(os.path.join(METRICS_COMPLEXITY_PATH, 'kmeans_labels.csv'))

    click.echo('Reducing dimensionality to 2D for visualization...')
    X = tsne_reduce_dim(X, 2)
    df_2d = pd.DataFrame(X, columns=['x', 'y'], index=df_features.index)
    df_2d['kmeans_label'] = df_features['kmeans_label']
    df_2d['complexity'] = df_features['complexity']

    df_2d.to_csv(os.path.join(METRICS_COMPLEXITY_PATH, 'complexity_clusters_2d.csv'))
    click.echo(f'Clustering written to "{METRICS_COMPLEXITY_PATH}"')


def visualize_clusters(quantile):
    """
    Visualize clusters of HTML page features.

    :param quantile: complexity quantile to align with cluster boundaries
    """

    p = os.path.join(METRICS_COMPLEXITY_PATH, 'complexity_quantiles.csv')
    if not os.path.isfile(p):
        raise click.FileError(p, 'Please calculate page complexities first.')
    quantiles = pd.read_csv(p, index_col=0)

    def binarize_complexity(x):
        x['complexity'] = int(x['complexity'] >= quantiles.loc[float(quantile)]['complexity'])
        return x

    in_path = os.path.join(METRICS_COMPLEXITY_PATH, 'complexity_clusters_2d.csv')
    if not os.path.isfile(in_path):
        raise click.FileError(p, 'Please calculate page complexities first.')

    df_2d = pd.read_csv(in_path, index_col='hash_key')
    df_2d = df_2d.apply(binarize_complexity, axis='columns')

    def sub_plt(ax, label_col, title, labels):

        for i, l in enumerate(labels):
            filtered = df_2d[df_2d[label_col] == i]
            ax.scatter(
                x=filtered['x'],
                y=filtered['y'],
                s=5,
                alpha=0.5,
                label=l
            )
        leg = ax.legend(loc='lower right', fontsize='small', borderpad=0.4, shadow=False)
        leg.get_frame().set_linewidth(0.0)
        ax.set_title(title, fontsize='medium')
        ax.spines['top'].set_visible(False)
        ax.set_xticks(ticks=np.linspace(*ax.get_xlim(), 5), labels=[])
        ax.set_yticks(ticks=np.linspace(*ax.get_ylim(), 5), labels=[])
        ax.spines['right'].set_visible(False)

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 2.5), dpi=200)
    sub_plt(ax1, 'kmeans_label', '$k$-Means Clustering', ['Cluster 0', 'Cluster 1'])
    sub_plt(ax2, 'complexity', 'Complexity Quantiles', ['Low', 'High'])

    plt.tight_layout(pad=0.5)
    plt.savefig(os.path.join(METRICS_COMPLEXITY_PATH, f'complexity_clusters_2d.png'))
    plt.savefig(os.path.join(METRICS_COMPLEXITY_PATH, f'complexity_clusters_2d.pdf'))

    click.echo(f'Plots written to "{METRICS_COMPLEXITY_PATH}')


def visualize_datasets(datasets):
    """
    Visualize median complexity of the datasets.

    :param datasets: list of dataset names
    """
    complexities = []
    with click.progressbar(datasets, label='Loading datasets') as progress:
        for ds in progress:
            path = os.path.join(METRICS_COMPLEXITY_PATH, ds, f'{ds}_complexity.csv')
            if not os.path.isfile(path):
                continue
            complexities.append(pd.read_csv(path)['complexity'])

    # Sort by median
    complexities, datasets = zip(*sorted(zip(complexities, datasets), key=lambda x: x[0].median(), reverse=True))

    plt.figure(figsize=(5, 3), dpi=200)
    plt.boxplot(
        complexities,
        positions=range(len(complexities)),
        labels=[DATASETS[d] for d in datasets],
    )

    plt.ylabel('Page Complexity')
    plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.ylim((-0.1, 1.1))
    for y in plt.gca().get_yticks():
        plt.axhline(y, linewidth=0.25, color='lightgrey', zorder=-1)

    plt.tight_layout()
    plt.savefig(os.path.join(METRICS_COMPLEXITY_PATH, f'complexity.png'))
    plt.savefig(os.path.join(METRICS_COMPLEXITY_PATH, f'complexity.pdf'))

    complexity_quantiles = pd.read_csv(os.path.join(METRICS_COMPLEXITY_PATH, 'complexity_quantiles.csv'), index_col=0)
    quantile_threshold = complexity_quantiles.loc[0.33]['complexity']

    # Sort back into alphabetical order
    complexities, datasets = zip(*sorted(zip(complexities, datasets), key=lambda x: x[1]))

    click.echo('Dataset stats:')
    click.echo('--------------')
    for i, compl in enumerate(complexities):
        click.echo(f'{datasets[i]:<20} ', nl=False)
        click.echo(f'pages: {compl.count():<10} ', nl=False)
        click.echo(f'pages low: {compl[compl < quantile_threshold].count():<10} ', nl=False)
        click.echo(f'pages high: {compl[compl >= quantile_threshold].count():<10} ', nl=False)
        click.echo(f'median complexity: {compl.median():.2f}', nl=False)
        click.echo()
    click.echo()

    click.echo(f'Complexity plots written to "{METRICS_COMPLEXITY_PATH}".')
