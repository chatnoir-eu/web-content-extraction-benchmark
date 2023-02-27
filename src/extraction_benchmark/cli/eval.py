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

import glob

import click
from extraction_benchmark.globals import *


@click.group()
def eval():
    """
    Evaluate model answers against the ground truth.
    """


@eval.command()
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
        dataset = sorted(DATASETS)
    if 'all' in model:
        model = sorted(MODELS)
    if eval_ensembles:
        model = sorted(m for m in MODELS if m.startswith('ensemble_'))
    metric = sorted(SCORES) if metric == 'all' else [metric]

    if not dataset:
        click.echo('No datasets selected.', err=True)
        return

    if not model:
        click.echo('No models selected.', err=True)
        return

    import nltk
    try:
        # Needed for Rouge
        nltk.data.find('tokenizers/punkt')
    except:
        nltk.download('punkt')

    from extraction_benchmark.eval import calculcate_scores
    try:
        calculcate_scores(metric, dataset, model, parallelism)
    except FileNotFoundError as e:
        click.FileError(e.filename,
                        f'Make sure you have converted the raw datasets using "convert-datasets".')


@eval.command()
@click.argument('score', type=click.Choice(['all', *SCORES]))
@click.option('-m', '--model', type=click.Choice(['all', *MODELS]), default=['all'], multiple=True)
@click.option('-d', '--dataset', type=click.Choice(['all', *DATASETS]), default=['all'], multiple=True)
@click.option('-x', '--exclude-dataset', type=click.Choice(DATASETS), default=[], multiple=True)
@click.option('-c', '--complexity', type=click.Choice(['all', *COMPLEXITIES]), default=['all'],
              required=True, multiple=True)
def aggregate(score, model, dataset, exclude_dataset, complexity):
    """
    Aggregate calculated performance metrics.
    """
    score = sorted(SCORES) if score == 'all' else [score]

    if 'all' in model:
        model = sorted(MODELS)
    if 'all' in dataset:
        dataset = sorted(d for d in DATASETS if d not in exclude_dataset)

    if not dataset:
        click.echo('No datasets selected.', err=True)
        return

    if not model:
        click.echo('No models selected.', err=True)
        return

    from extraction_benchmark.eval import aggregate_scores
    try:
        with click.progressbar(score, label='Aggregating scores') as progress:
            for s in progress:
                aggregate_scores(s, model, dataset, complexity)
    except FileNotFoundError as e:
        raise click.FileError(e.filename, 'Please calculate complexity scores first.')

    click.echo(f'Aggregation written to "{METRICS_PATH}"')


@eval.command()
def cythonize_rouge():
    """
    Cythonize Rouge-Score module.

    By cythonizing the Rouge-Score module, the slow scoring performance can be improved slightly.
    """

    click.confirm('This will cythonize the rouge-score module. '
                  'You will have to reinstall it to revert the changes. Continue?', abort=True)

    import rouge_score
    path = os.path.dirname(rouge_score.__file__)

    py_files = glob.glob(os.path.join(path, '*.py'))
    if not py_files:
        click.echo('No Python files found in module. Has the module already been cythonized?', err=True)
        return

    for p in py_files:
        os.rename(p, p + 'x')

    from Cython.Build.Cythonize import main as cython_main
    cython_main(['cythonize', '-3', '--inplace', *glob.glob(os.path.join(path, '*.pyx'))])
