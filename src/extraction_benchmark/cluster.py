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

from collections import defaultdict
from multiprocessing import Pool
import re
import click
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from resiliparse.parse.html import HTMLTree

from extraction_benchmark.dataset_readers import read_dataset
from extraction_benchmark.extract import DATASETS
from extraction_benchmark.paths import *


_TOKEN_RE = re.compile(r'\w+', flags=re.UNICODE)
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

    features['html_to_non_html'] = n_tags / len(_TOKEN_RE.findall(text))

    return features


def calculate_dataset_features(dataset):
    df = pd.DataFrame()
    df.index.name = 'hash_key'
    for hash_key, data in read_dataset(dataset, False):
        features = extract_html_features(data['articleBody'])
        s = pd.Series(features, name=hash_key)
        df = df.append(s)
    df.to_csv(os.path.join(DATASET_TRUTH_PATH, dataset, f'{dataset}_html_features.csv'))
    return ''


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


@click.group()
def cluster():
    pass


@cluster.command()
@click.option('-d', '--dataset', type=click.Choice(['all', *DATASETS]), default=['all'], multiple=True)
@click.option('-p', '--parallelism', help='Number of threads to use', default=os.cpu_count())
def extract_features(dataset, parallelism):
    if 'all' in dataset:
        dataset = DATASETS.keys()

    with Pool(processes=parallelism) as pool:
        for _ in tqdm(pool.imap_unordered(calculate_dataset_features, dataset),
                      total=len(dataset), desc='Extracting dataset features'):
            pass


@cluster.command()
@click.option('-d', '--dataset', type=click.Choice(['all', *DATASETS]), default=['all'], multiple=True)
@click.option('-r', '--reduce-dim', type=int,
              help='Reduce dimensionality before clustering (0 for no reduction)')
@click.option('-c', '--clusters', type=int, default=2, help='Number of clusters')
def kmeans(dataset, reduce_dim, clusters):
    if 'all' in dataset:
        dataset = DATASETS

    df_features = pd.DataFrame()
    df_complexity = pd.DataFrame()
    for ds in tqdm(dataset, desc='Loading datasets', leave=False):
        df_tmp = pd.read_csv(os.path.join(DATASET_TRUTH_PATH, ds, f'{ds}_html_features.csv'))
        df_tmp['dataset'] = ds
        df_features = df_features.append(df_tmp, ignore_index=True)

        df_tmp = pd.read_csv(os.path.join(DATASET_TRUTH_PATH, ds, f'{ds}_complexity.csv'))
        df_tmp['dataset'] = ds
        df_complexity = df_complexity.append(df_tmp, ignore_index=True)

    df_features.set_index(['hash_key', 'dataset'], inplace=True)
    df_complexity.set_index(['hash_key', 'dataset'], inplace=True)

    df_features = df_features.join(df_complexity, how='inner')

    scaler = StandardScaler()
    X = scaler.fit_transform(df_features.drop(columns='complexity'))

    if reduce_dim:
        X = pca_reduce_dim(X, reduce_dim)

    click.echo('Clustering datapoints...')
    labels = get_kmeans_labels(X, clusters)
    # Ensure cluster labels are aligned with quantiles
    if sum(labels[labels == 1]) < len(labels[labels == 0]):
        labels = 1 - labels
    df_features['kmeans_label'] = labels
    df_features.to_csv(os.path.join(DATASET_TRUTH_PATH, 'kmeans_labels.csv'))

    click.echo('Reducing dimensionality to 2D for visualization...')
    X = tsne_reduce_dim(X, 2)
    df_2d = pd.DataFrame(X, columns=['x', 'y'], index=df_features.index)
    df_2d['kmeans_label'] = df_features['kmeans_label']
    df_2d['complexity'] = df_features['complexity']

    df_2d.to_csv(os.path.join(DATASET_TRUTH_PATH, 'complexity_clusters_2d.csv'))
    click.echo(f'Clustering written to "{DATASET_TRUTH_PATH}"')


@cluster.command()
@click.option('-q', '--quantile', type=click.Choice(['0.25', '0.33', '0.5', '0.66', '0.75']), default='0.33',
              help='Quantile boundary')
def visualize(quantile):
    # Map complexity scores to quantiles
    quantiles = pd.read_csv(os.path.join(DATASET_TRUTH_PATH, 'complexity_quantiles.csv'), index_col=0)

    def binarize_complexity(x):
        x['complexity'] = int(x['complexity'] >= quantiles.loc[float(quantile)]['complexity'])
        return x

    in_path = os.path.join(DATASET_TRUTH_PATH, 'complexity_clusters_2d.csv')
    if not os.path.isfile(in_path):
        raise click.UsageError('Calculate clusters with the "cluster" subcommand first.')

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
    plt.savefig(os.path.join(DATASET_TRUTH_PATH, f'complexity_clusters_2d.png'))
    plt.savefig(os.path.join(DATASET_TRUTH_PATH, f'complexity_clusters_2d.pdf'))

    click.echo(f'Plots written to "{DATASET_TRUTH_PATH}')
