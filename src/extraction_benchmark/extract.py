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
import gzip
import os
from functools import partial
from itertools import product
import json
from multiprocessing import get_context
from threading import Thread
import warnings

import click

from extraction_benchmark.dataset_readers import read_dataset
from extraction_benchmark.extractors import extractors
from extraction_benchmark.paths import *


def _dump_json(filepath, extracted):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(extracted, f, indent=4, ensure_ascii=False)


def extract_ground_truth(datasets):
    """
    Convert ground truth from raw dataset to JSON format.

    :param datasets: list of input dataset
    """
    for ds in datasets:
        with click.progressbar(read_dataset(ds, True), label=f'Converting truth of {ds}') as ds_progress:
            extracted = {k: v for k, v in ds_progress}

        _dump_json(os.path.join(DATASET_TRUTH_PATH, ds, ds + '.json'), extracted)


def extract_raw_html(datasets):
    """
    Convert HTML files from raw dataset to JSON format.

    :param datasets: list of input dataset
    """
    for ds in datasets:
        out_dir = os.path.join(DATASET_HTML_PATH, ds)
        os.makedirs(out_dir, exist_ok=True)
        with click.progressbar(read_dataset(ds, False), label=f'Converting HTML of {ds}') as ds_progress:
            for page_id, val in ds_progress:
                if not val.get('html'):
                    continue
                with gzip.GzipFile(os.path.join(out_dir, page_id + '.html.gz'), 'w') as f:
                    f.write(val['html'].encode())


def _extract_with_model_expand_args(args, skip_existing=False):
    _extract_with_model(*args, skip_existing=skip_existing)


def _extract_with_model(model, dataset, skip_existing=False):
    model, model_name = model
    out_path = os.path.join(MODEL_OUTPUTS_PATH, dataset, model_name, model_name + '.json')

    extracted = {}
    if skip_existing and os.path.isfile(out_path):
        extracted = json.load(open(out_path, 'r'))

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        for file_hash, data in read_dataset(dataset, False):
            if file_hash in extracted:
                continue

            text = data['articleBody']
            data['articleBody'] = ''

            def _model_wrapper():
                try:
                    data['articleBody'] = model(text, page_id=file_hash) or ''
                except Exception as e:
                    click.echo(f'Error in model {model_name}: {str(e)}', err=True)

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


def extract(models, datasets, skip_existing, parallelism):
    """
    Extract datasets with the selected extraction models.

    :param models: list of extraction model names (if ``ground_truth == False``)
    :param datasets: list of dataset names under "datasets/raw"
    :param skip_existing: skip models for which an answer file exists already
    :param parallelism: number of parallel workers
    """

    if ('web2text' in models or 'boilernet' in models) and parallelism > 1:
        click.echo('WARNING: Deep neural models should be run separately and with --parallelism=1', err=True)

    model = [(getattr(extractors, 'extract_' + m), m) for m in models]
    jobs = list(product(model, datasets))

    if parallelism == 1:
        with click.progressbar(jobs, label='Running extrators') as progress:
            for job in progress:
                _extract_with_model_expand_args(job)
        return

    with get_context('spawn').Pool(processes=parallelism) as pool:
        try:
            with click.progressbar(pool.imap_unordered(partial(_extract_with_model_expand_args,
                                                               skip_existing=skip_existing), jobs),
                                   length=len(jobs), label='Running extrators') as progress:
                for _ in progress:
                    pass
        except KeyboardInterrupt:
            pool.terminate()

    click.echo(f'Model outputs written to {MODEL_OUTPUTS_PATH}')
