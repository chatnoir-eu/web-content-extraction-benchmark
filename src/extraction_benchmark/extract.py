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
import os
from contextlib import redirect_stderr, redirect_stdout
from functools import partial
import io
from itertools import product
import json
import logging
from multiprocessing import get_context
from threading import Thread
from typing import Any, Dict
import warnings

import click
import lz4.frame

from extraction_benchmark.dataset_readers import read_datasets, read_raw_dataset
from extraction_benchmark.extractors import extractors
from extraction_benchmark.paths import *


def _dict_to_jsonl(filepath, lines_dict: Dict[str, Any]):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        for page_id in sorted(lines_dict):
            json.dump({'page_id': page_id, **lines_dict[page_id]}, f, indent=None, ensure_ascii=False)
            f.write('\n')


def extract_ground_truth(datasets):
    """
    Convert ground truth from raw dataset to JSON format.

    :param datasets: list of input dataset
    :return: set of page IDs that were extracted
    """

    page_ids = set()
    for ds in datasets:
        with click.progressbar(read_raw_dataset(ds, True), label=f'Converting ground truth of {ds}') as ds_progress:
            extracted = {k: v for k, v in ds_progress}
            page_ids.update(extracted.keys())
        _dict_to_jsonl(os.path.join(DATASET_COMBINED_TRUTH_PATH, f'{ds}.jsonl'), extracted)
    return page_ids


def extract_raw_html(datasets, page_id_whitelist=None):
    """
    Convert HTML files from raw dataset to JSON format.

    :param datasets: list of input dataset
    :param page_id_whitelist: optional list of page IDs to include (if set, IDs not in this list will be skipped)
    """
    if page_id_whitelist and type(page_id_whitelist) is not set:
        page_id_whitelist = set(page_id_whitelist)

    for ds in datasets:
        out_dir = os.path.join(DATASET_COMBINED_HTML_PATH, ds)
        os.makedirs(out_dir, exist_ok=True)
        with click.progressbar(read_raw_dataset(ds, False), label=f'Converting HTML of {ds}') as ds_progress:
            for page_id, val in ds_progress:
                if page_id_whitelist and page_id not in page_id_whitelist:
                    continue
                if not val.get('html'):
                    continue
                with open(os.path.join(out_dir, page_id + '.html.lz4'), 'wb') as f:
                    f.write(lz4.frame.compress(val['html'].encode()))


def _extract_with_model_expand_args(args, skip_existing=False, verbose=False):
    _extract_with_model(*args, skip_existing=skip_existing, verbose=verbose)


def _extract_with_model(model, dataset, skip_existing=False, verbose=False):
    model, model_name = model
    out_path = os.path.join(MODEL_OUTPUTS_PATH, dataset, model_name + '.jsonl')

    logger = logging.getLogger('wceb-extract')
    logger.setLevel(logging.INFO if verbose else logging.ERROR)

    extracted = {}
    if skip_existing and os.path.isfile(out_path):
        with open(out_path, 'r') as f:
            for line in f:
                j = json.loads(line)
                extracted[j['page_id']] = {k: v for k, v in j.items() if k != 'page_id'}

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        for file_hash, in_data in read_datasets([dataset], False):
            if file_hash in extracted:
                continue

            out_data = dict(plaintext='', model=model_name)

            def _model_wrapper():
                try:
                    with redirect_stdout(io.StringIO()) as stdout, redirect_stderr(io.StringIO()) as stderr:
                        out_data['plaintext'] = model(in_data['html'], page_id=file_hash) or ''

                    if stdout.getvalue():
                        logger.info(stdout.getvalue().strip())
                    if stderr.getvalue():
                        logger.warning(stderr.getvalue().strip())
                except Exception as e:
                    logger.warning(f'Error in model {model_name} while extracting {dataset}:{file_hash}:')
                    logger.warning(str(e))

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

            extracted[file_hash] = out_data

    if not extracted:
        return

    _dict_to_jsonl(out_path, extracted)


def extract(models, datasets, skip_existing, parallelism, verbose=False):
    """
    Extract datasets with the selected extraction models.

    :param models: list of extraction model names (if ``ground_truth == False``)
    :param datasets: list of dataset names under "datasets/raw"
    :param skip_existing: skip models for which an answer file exists already
    :param parallelism: number of parallel workers
    :param verbose: log error information
    """

    model = [(getattr(extractors, 'extract_' + m), m) for m in models]
    jobs = list(product(model, datasets))

    def item_show_func(j):
        if j:
            return f'Model: {j[0][1]}, Dataset: {j[1]}'

    if parallelism == 1:
        with click.progressbar(jobs, label='Running extrators', item_show_func=item_show_func) as progress:
            for job in progress:
                _extract_with_model_expand_args(job)
        return

    with get_context('spawn').Pool(processes=parallelism) as pool:
        try:
            with click.progressbar(pool.imap_unordered(partial(_extract_with_model_expand_args,
                                                               skip_existing=skip_existing, verbose=verbose), jobs),
                                   length=len(jobs), label='Running extrators') as progress:
                for _ in progress:
                    pass
        except KeyboardInterrupt:
            pool.terminate()

    click.echo(f'Model outputs written to {MODEL_OUTPUTS_PATH}')
