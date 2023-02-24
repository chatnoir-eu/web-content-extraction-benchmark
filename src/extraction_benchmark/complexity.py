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

import re

import click
from matplotlib import pyplot as plt
import pandas as pd
from resiliparse.parse.html import HTMLTree
from tqdm import tqdm

from extraction_benchmark.extract import read_dataset
from extraction_benchmark.paths import *
from extraction_benchmark.extract import DATASETS


@click.group()
def complexity():
    pass


_COMPLEXITY_METRICS_PATH = os.path.join(METRICS_PATH, 'complexity')
_TOKEN_RE = re.compile(r'\w+', flags=re.UNICODE)


def _tokenize(text):
    return _TOKEN_RE.findall(text)


@complexity.command()
@click.option('-d', '--dataset', type=click.Choice(['all', *DATASETS]), default=['all'], multiple=True)
def calculate(dataset):
    if 'all' in dataset:
        dataset = list(DATASETS.keys())

    if not dataset:
        return

    complexity_total = pd.DataFrame(columns=['complexity'])
    complexity_total.index.name = 'hash_key'
    quantile_labels = [0.25, 0.33, 0.5, 0.66, 0.75]

    os.makedirs(_COMPLEXITY_METRICS_PATH, exist_ok=True)

    for ds in tqdm(dataset, desc='Iterating datasets', leave=False):
        tokens_truth = {}
        tokens_src = {}
        for h, truth in tqdm(read_dataset(ds, True), desc=f'Reading truth files ({ds})', leave=False):
            tokens_truth[h] = len(_tokenize(truth['articleBody']))
        for h, src in tqdm(read_dataset(ds, False), desc=f'Reading source files ({ds})', leave=False):
            if h not in tokens_truth:
                continue
            # Extract all text tokens except script / style
            tree = HTMLTree.parse(src['articleBody'])
            for e in tree.body.query_selector_all('script, style'):
                e.decompose()
            tokens_src[h] = len(_tokenize(tree.body.text))

        tokens_truth = pd.DataFrame.from_dict(tokens_truth, orient='index')
        tokens_src = pd.DataFrame.from_dict(tokens_src, orient='index')

        out_path_ds = os.path.join(_COMPLEXITY_METRICS_PATH, ds)
        os.makedirs(out_path_ds, exist_ok=True)

        complexity = 1 - (tokens_truth / tokens_src).clip(lower=0, upper=1)
        complexity.index.name = 'hash_key'
        complexity.columns = ['complexity']
        complexity.to_csv(os.path.join(out_path_ds, f'{ds}_complexity.csv'))
        complexity['dataset'] = ds

        quantiles = complexity.quantile(quantile_labels)
        quantiles.to_csv(os.path.join(out_path_ds, f'{ds}_complexity_quantiles.csv'))

        complexity_total = complexity_total.append(complexity)

    complexity_total.reset_index(inplace=True)
    complexity_total.set_index(['hash_key', 'dataset'], inplace=True)
    quantiles = complexity_total.quantile(quantile_labels)
    quantiles.to_csv(os.path.join(_COMPLEXITY_METRICS_PATH, f'complexity_quantiles.csv'))
    complexity_total.to_csv(os.path.join(_COMPLEXITY_METRICS_PATH, f'complexity.csv'))

    click.echo(f'Complexity scores written to "{_COMPLEXITY_METRICS_PATH}".')


@complexity.command()
@click.option('-d', '--dataset', type=click.Choice(['all', *DATASETS]), default=['all'], multiple=True)
def visualize(dataset):
    if 'all' in dataset:
        dataset = list(DATASETS.keys())

    complexities = []
    for ds in tqdm(dataset, desc='Iterating datasets', leave=False):
        path = os.path.join(_COMPLEXITY_METRICS_PATH, ds, f'{ds}_complexity.csv')
        if not os.path.isfile(path):
            continue
        complexities.append(pd.read_csv(path)['complexity'])

    # Sort by median
    complexities, dataset = zip(*sorted(zip(complexities, dataset), key=lambda x: x[0].median(), reverse=True))

    plt.figure(figsize=(5, 3), dpi=200)
    plt.boxplot(
        complexities,
        positions=range(len(complexities)),
        labels=[DATASETS[d] for d in dataset],
    )

    plt.ylabel('Page Complexity')
    plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.ylim((-0.1, 1.1))
    for y in plt.gca().get_yticks():
        plt.axhline(y, linewidth=0.25, color='lightgrey', zorder=-1)

    plt.tight_layout()
    plt.savefig(os.path.join(_COMPLEXITY_METRICS_PATH, f'complexity.png'))
    plt.savefig(os.path.join(_COMPLEXITY_METRICS_PATH, f'complexity.pdf'))

    complexity_quantiles = pd.read_csv(os.path.join(_COMPLEXITY_METRICS_PATH, 'complexity_quantiles.csv'), index_col=0)
    quantile_threshold = complexity_quantiles.loc[0.33]['complexity']

    # Sort back into alphabetical order
    complexities, dataset = zip(*sorted(zip(complexities, dataset), key=lambda x: x[1]))

    click.echo('Dataset stats:')
    click.echo('--------------')
    for i, compl in enumerate(complexities):
        click.echo(f'{dataset[i]:<20} ', nl=False)
        click.echo(f'pages: {compl.count():<10} ', nl=False)
        click.echo(f'pages low: {compl[compl < quantile_threshold].count():<10} ', nl=False)
        click.echo(f'pages high: {compl[compl >= quantile_threshold].count():<10} ', nl=False)
        click.echo(f'median complexity: {compl.median():.2f}', nl=False)
        click.echo()
    click.echo()

    click.echo(f'Complexity plots written to "{_COMPLEXITY_METRICS_PATH}".')
