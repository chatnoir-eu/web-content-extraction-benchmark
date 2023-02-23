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

import ctypes
from functools import partial
from itertools import product
import json
from multiprocessing import get_context
from threading import Thread

import click
from tqdm import tqdm

from extraction_benchmark import extractors
from extraction_benchmark.dataset_readers import read_dataset
from extraction_benchmark.extractors import list_extractors
from extraction_benchmark.paths import *


DATASETS = {
    'cetd': 'CETD',
    'cleaneval': 'CleanEval',
    'cleanportaleval': 'CleanPortalEval',
    'dragnet': 'Dragnet',
    'google-trends-2017': 'Google-Trends',
    'l3s-gn1': 'L3S-GN1',
    'readability': 'Readability',
    'scrapinghub': 'ScrapingHub'
}

MODELS = list_extractors(include_ensembles=True)


def _dump_json(filepath, extracted):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(extracted, f, indent=4, ensure_ascii=False)


def _extract_with_model_expand_args(args, skip_existing=False):
    _extract_with_model(*args, skip_existing=skip_existing)


def _extract_with_model(model, dataset, skip_existing=False):
    model, model_name = model
    out_path = os.path.join(MODEL_OUTPUTS_PATH, dataset, model_name, model_name + '.json')

    extracted = {}
    if skip_existing and os.path.isfile(out_path):
        extracted = json.load(open(out_path, 'r'))

    for file_hash, data in read_dataset(dataset, False):
        if file_hash in extracted:
            continue

        text = data['articleBody']
        data['articleBody'] = ''

        def _model_wrapper():
            try:
                data['articleBody'] = model(text, page_id=file_hash) or ''
            except:
                pass

        if model.__name__.startswith('extract_ensemble_'):
            # Threading not needed for ensemble and only creates problems
            _model_wrapper()
        else:
            t = Thread(target=_model_wrapper)
            t.start()
            t.join(timeout=30)
            if t.is_alive():
                # Kill hanging thread
                ctypes.pythonapi.PyThreadState_SetAsyncExc(t.ident, ctypes.py_object(Exception))

        extracted[file_hash] = {**data, 'model': model_name}

    if not extracted:
        return

    _dump_json(out_path, extracted)


def _extract_ground_truth(input_dataset):
    ds = tqdm(read_dataset(input_dataset, True), desc='Extracting truth of ' + input_dataset, leave=False)
    extracted = {k: v for k, v in ds}
    _dump_json(os.path.join(DATASET_TRUTH_PATH, input_dataset, input_dataset + '.json'), extracted)


@click.group()
def extract():
    pass


@extract.command()
@click.option('-m', '--model', type=click.Choice(['all', *MODELS]), default=[], multiple=True)
@click.option('--run-ensembles',is_flag=True, help='Run all ensembles')
@click.option('-e', '--exclude-model', type=click.Choice(MODELS), default=[], multiple=True)
@click.option('-d', '--dataset', type=click.Choice(['all', *DATASETS]), default=['all'], multiple=True)
@click.option('-x', '--exclude-dataset', type=click.Choice(DATASETS), default=[], multiple=True)
@click.option('-t', '--truth', is_flag=True, help='Extract ground truth')
@click.option('-s', '--skip-existing', is_flag=True, help='Load existing answer and extract only new')
@click.option('-p', '--parallelism', help='Number of threads to use', default=os.cpu_count())
def extract(model, run_ensembles, exclude_model, dataset, exclude_dataset, truth, skip_existing, parallelism):
    if 'all' in model:
        model = [m for m in MODELS if m not in exclude_model and not m.startswith('ensemble_')]
    if run_ensembles:
        model = [m for m in MODELS if m.startswith('ensemble_')]
    if 'all' in dataset:
        dataset = [d for d in DATASETS if d not in exclude_dataset]

    if truth:
        for ds in dataset:
            _extract_ground_truth(ds)
        return

    if ('web2text' in model or 'boilernet' in model) and parallelism > 1:
        click.echo('WARNING: Deep neural models should be run separately and with --parallelism=1', err=True)

    model = [(getattr(extractors, 'extract_' + m), m) for m in model]
    jobs = list(product(model, dataset))

    if parallelism == 1:
        for job in tqdm(jobs, desc='Running extrators'):
            _extract_with_model_expand_args(job)
        return

    with get_context('spawn').Pool(processes=parallelism) as pool:
        try:
            for _ in tqdm(pool.imap_unordered(partial(_extract_with_model_expand_args,
                                                      skip_existing=skip_existing), jobs),
                          total=len(jobs), desc='Running extrators'):
                pass
        except KeyboardInterrupt:
            pool.terminate()
