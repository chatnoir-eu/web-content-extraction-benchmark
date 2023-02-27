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


import sys

import click
from extraction_benchmark.globals import *


@click.group()
@click.pass_context
@click.argument('model', type=click.Choice(['all', *MODELS]), required=True, nargs=-1)
@click.option('--run-ensembles', is_flag=True, help='Run all ensembles')
@click.option('-e', '--exclude-model', type=click.Choice(MODELS), default=[], multiple=True)
@click.option('-d', '--dataset', type=click.Choice(['all', *DATASETS]), default=['all'], multiple=True)
@click.option('-x', '--exclude-dataset', type=click.Choice(DATASETS), default=[], multiple=True)
@click.option('-s', '--skip-existing', is_flag=True, help='Load existing answer and extract only new')
@click.option('-p', '--parallelism', help='Number of threads to use', default=os.cpu_count())
def extract(ctx, model, run_ensembles, exclude_model, dataset, exclude_dataset, skip_existing, parallelism):
    """
    Run main content extractors on the datasets.
    """
    if 'all' in model:
        model = sorted(m for m in MODELS if m not in exclude_model)
    if run_ensembles:
        model = sorted(m for m in MODELS_ALL if m.startswith('ensemble_'))
    if 'all' in dataset:
        dataset = sorted(d for d in DATASETS if d not in exclude_dataset)

    if not model:
        click.echo(f'No models selected. Run {sys.argv[0]} {ctx.command.name} --help for more info.', err=True)
        return

    if not dataset:
        click.echo('No input datasets found.\n'
                   'Make sure that all datasets have been extracted correctly to a folder "datasets/raw" '
                   'under the current working directory.', err=False)
        return

    from extraction_benchmark import extract
    try:
        extract.extract(model, dataset, skip_existing, parallelism)
    except FileNotFoundError as e:
        click.FileError(e.filename,
                        'Make sure that all datasets have been extracted correctly to a folder "datasets/raw" '
                        'under the current working directory.')


@click.group()
def convert():
    """
    Convert raw datasets to JSON format.
    """


@convert.command()
@click.option('-d', '--dataset', type=click.Choice(['all', *DATASETS]), default=['all'], multiple=True)
@click.option('-x', '--exclude-dataset', type=click.Choice(DATASETS), default=[], multiple=True)
def convert_truth(dataset, exclude_dataset):
    """
    Convert raw ground truth to JSON format.
    """
    if 'all' in dataset:
        dataset = sorted(d for d in DATASETS if d not in exclude_dataset)

    from extraction_benchmark import extract
    try:
        extract.extract_ground_truth(dataset)
    except FileNotFoundError as e:
        click.FileError(e.filename,
                        'Make sure that all datasets have been extracted correctly to a folder "datasets/raw" '
                        'under the current working directory.')


@convert.command()
@click.option('-d', '--dataset', type=click.Choice(['all', *DATASETS]), default=['all'], multiple=True)
@click.option('-x', '--exclude-dataset', type=click.Choice(DATASETS), default=[], multiple=True)
def convert_html(dataset, exclude_dataset):
    """
    Convert raw HTML pages to JSON format.
    """
    if 'all' in dataset:
        dataset = sorted(d for d in DATASETS if d not in exclude_dataset)

    from extraction_benchmark import extract
    try:
        extract.extract_raw_html(dataset)
    except FileNotFoundError as e:
        click.FileError(e.filename,
                        'Make sure that all datasets have been extracted correctly to a folder "datasets/raw" '
                        'under the current working directory.')
