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
def eval():
    """
    Evaluate model answers against the ground truth.
    """


@eval.command(name='score')
@click.argument('metric', type=click.Choice(['all', *SCORES]))
@click.option('-d', '--dataset', type=click.Choice(['all', *DATASETS]), default=['all'], multiple=True)
@click.option('-m', '--model', type=click.Choice(['all', *MODELS]), default=['all'], multiple=True)
@click.option('--eval-ensembles', is_flag=True)
@click.option('-p', '--parallelism', help='Number of threads to use', default=os.cpu_count())
def score(metric, dataset, model, eval_ensembles, parallelism):
    """
    Calculate performance metrics on model answers.
    """

    if 'all' in dataset:
        dataset = DATASETS

    if 'all' in model:
        model = MODELS
    if eval_ensembles:
        model = [m for m in MODELS if m.startswith('ensemble_')]
    metric = SCORES if metric == 'all' else [metric]

    if not dataset:
        click.echo('No datasets selected.', err=True)
        return

    if not model:
        click.echo('No models selected.', err=True)
        return

    from extraction_benchmark.eval import calculcate_scores
    calculcate_scores(metric, dataset, model, parallelism)


@eval.command(name='aggregate')
@click.argument('score', type=click.Choice(SCORES))
@click.option('-m', '--model', type=click.Choice(['all', *MODELS]), default=['all'], multiple=True)
@click.option('-d', '--dataset', type=click.Choice(['all', *DATASETS]), default=['all'], multiple=True)
@click.option('-x', '--exclude-dataset', type=click.Choice(DATASETS), default=[], multiple=True)
@click.option('-c', '--complexity', type=click.Choice(['all', *COMPLEXITIES]), default=['all'],
              required=True, multiple=True)
def aggregate(score, model, dataset, exclude_dataset, complexity):
    """
    Aggregate calculated performance metrics.
    """
    if 'all' in model:
        model = MODELS
    if 'all' in dataset:
        dataset = [d for d in DATASETS if d not in exclude_dataset]

    if not dataset:
        click.echo('No datasets selected.', err=True)
        return

    if not model:
        click.echo('No models selected.', err=True)
        return

    from extraction_benchmark.eval import aggregate_scores
    try:
        aggregate_scores(score, model, dataset, complexity)
    except FileNotFoundError as e:
        raise click.FileError(e.filename, 'Please calculate complexity scores first.')
