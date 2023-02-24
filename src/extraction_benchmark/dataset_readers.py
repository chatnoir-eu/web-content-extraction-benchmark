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

import gzip
import hashlib
import json
import re

from resiliparse.parse import bytes_to_str, detect_encoding
from resiliparse.parse.html import HTMLTree, NodeType

from extraction_benchmark.paths import *


def _hash(data):
    m = hashlib.sha256()
    m.update(data)
    return m.hexdigest()


def _file_hash(file):
    with open(file, 'rb') as f:
        return _hash(f.read())


def _extract_dict(source_dataset, source_case, content, is_truth=False, **kwargs):
    d = {
        ('plaintext' if is_truth else 'html'): content,
        **{k: v for k, v in kwargs.items() if v},
        'source': [source_dataset, source_case] if source_case else [source_dataset]
    }
    return d


def _read_file(path):
    with open(path, 'rb') as f:
        file_bytes = f.read()
        if path.endswith('.gz'):
            file_bytes = gzip.decompress(file_bytes)
        enc = detect_encoding(file_bytes, max_len=100000, html5_compatible=False) or 'utf-8'
        return bytes_to_str(file_bytes, encoding=enc, fallback_encodings=['utf-8', 'cp1252'])


def read_cleaneval(ground_truth=False, portal=False):
    if not portal:
        # Original CleanEval
        dataset_path = os.path.join(DATASET_RAW_PATH, 'cleaneval', 'orig')
        dataset_path_truth = os.path.join(DATASET_RAW_PATH, 'cleaneval', 'clean')
    else:
        # CleanPortalEval
        dataset_path = os.path.join(DATASET_RAW_PATH, 'cleanportaleval', 'input')
        dataset_path_truth = os.path.join(DATASET_RAW_PATH, 'cleanportaleval', 'GoldStandard')

    read_path = dataset_path_truth if ground_truth else dataset_path

    text_tag_re = re.compile(r'(?:^<text [^>]+>\s*|\s*</text>$)', flags=re.MULTILINE)

    for file in os.listdir(read_path):
        abs_path = os.path.join(read_path, file)
        content = _read_file(abs_path)
        url = None
        if ground_truth:
            url = re.search(r'^\s*URL: (https?://.+)', content)
            if url:
                url = url.group(1)
            content = HTMLTree.parse(content).body.text
            content = re.sub(r'\n +', '\n', content)
            content = re.sub(r'^\s*URL:[^\n]+\s*', '', content)    # Strip URL line

        if ground_truth:
            abs_path = os.path.join(dataset_path, os.path.splitext(file)[0] + '.html')
        else:
            content = text_tag_re.sub('', content)
        source = os.path.splitext(file)[0]
        yield _file_hash(abs_path), _extract_dict('cleaneval', source, content, is_truth=ground_truth, url=url)


def read_cleanportaleval(ground_truth=False):
    return read_cleaneval(ground_truth, True)


def read_dragnet(ground_truth=False):
    dataset_path = os.path.join(DATASET_RAW_PATH, 'dragnet', 'HTML')
    dataset_path_truth = os.path.join(DATASET_RAW_PATH, 'dragnet', 'corrected', 'Corrected')
    read_path = dataset_path_truth if ground_truth else dataset_path

    for file in os.listdir(read_path):
        abs_path = os.path.join(read_path, file)
        content = _read_file(abs_path)
        if ground_truth:
            file = os.path.splitext(os.path.splitext(file)[0])[0]
            abs_path = os.path.join(dataset_path, file)
        source = os.path.splitext(file)[0]
        yield _file_hash(abs_path), _extract_dict('dragnet', source, content, is_truth=ground_truth)


def read_cetd(ground_truth=False):
    dataset_path = os.path.join(DATASET_RAW_PATH, 'cetd')
    verticals = ['arstechnica', 'BBC', 'Chaos', 'nytimes', 'wiki', 'YAHOO!']
    for vertical in verticals:
        sub_path = os.path.join(dataset_path, vertical, 'gold' if ground_truth else 'original')
        for file in os.listdir(sub_path):
            abs_path = os.path.join(sub_path, file)
            content = _read_file(abs_path)
            if ground_truth:
                abs_path = os.path.join(dataset_path, vertical, 'original', os.path.splitext(file)[0] + '.htm')
            source = vertical + '_' + os.path.splitext(file)[0]
            yield _file_hash(abs_path), _extract_dict('cetd', source, content, is_truth=ground_truth)


def read_readability(ground_truth=False):
    dataset_path = os.path.join(DATASET_RAW_PATH, 'readability', 'test-pages')
    for case_dir in os.listdir(dataset_path):
        sub_path = 'expected.html' if ground_truth else 'source.html'
        abs_path = os.path.join(dataset_path, os.path.join(case_dir, sub_path))
        content = _read_file(abs_path)
        if ground_truth:
            content = HTMLTree.parse(content).body.text
            abs_path = os.path.join(dataset_path, os.path.join(case_dir, 'source.html'))
        yield _file_hash(abs_path), _extract_dict('readability', case_dir, content, is_truth=ground_truth)


def read_scrapinghub_benchmark(ground_truth=False):
    dataset_path = os.path.join(DATASET_RAW_PATH, 'scrapinghub')

    if ground_truth:
        truth_json = json.load(open(os.path.join(dataset_path, 'ground-truth.json'), 'r'))
        for k, v in truth_json.items():
            yield k, _extract_dict('scrapinghub', None, v['articleBody'], is_truth=ground_truth, url=v['url'])
        return

    dataset_path = os.path.join(dataset_path, 'html')
    for file in os.listdir(dataset_path):
        abs_path = os.path.join(dataset_path, file)
        hash_id = os.path.splitext(os.path.splitext(file)[0])[0]
        with gzip.GzipFile(abs_path, 'r') as f:
            file_hash = _hash(f.read())
        yield file_hash, _extract_dict('scrapinghub', hash_id, _read_file(abs_path), is_truth=ground_truth)


def _extract_with_css_selector(html, selector):
    tree = HTMLTree.parse(html)
    elements = tree.body.query_selector_all(selector)
    content = ''
    for e in elements:
        if len(e.child_nodes) != 1 or e.first_child.type != NodeType.TEXT:
            # Only count leaf nodes to avoid adding an element multiple times
            continue
        if e.parent.tag in ['address', 'article', 'aside', 'blockquote', 'canvas', 'dd', 'div', 'dl', 'dt', 'fieldset',
                            'figcaption', 'figure', 'footer', 'form', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'header',
                            'hr', 'li', 'main', 'nav', 'noscript', 'ol', 'p', 'pre', 'section', 'table', 'tfoot',
                            'ul', 'video']:
            content += '\n'
        content += e.text.strip() + ' '
    return content.strip()


def read_l3s_gn1(ground_truth=False):
    dataset_path = os.path.join(DATASET_RAW_PATH, 'l3s-gn1', 'original')
    dataset_path_truth = os.path.join(DATASET_RAW_PATH, 'l3s-gn1', 'annotated')
    read_path = dataset_path_truth if ground_truth else dataset_path

    for file in os.listdir(read_path):
        abs_path = os.path.join(read_path, file)
        content = _read_file(abs_path)
        if ground_truth:
            abs_path = os.path.join(dataset_path, file)
            content = _extract_with_css_selector(content, '.x-nc-sel1, .x-nc-sel2, .x-nc-sel3')
        source = os.path.splitext(file)[0]
        yield _file_hash(abs_path), _extract_dict('l3s-gn1', source, content, is_truth=ground_truth)


def read_google_trends_2017(ground_truth=False):
    dataset_path = os.path.join(DATASET_RAW_PATH, 'google-trends-2017', 'raw_html')
    dataset_path_truth = os.path.join(DATASET_RAW_PATH, 'google-trends-2017', 'prepared_html')
    read_path = dataset_path_truth if ground_truth else dataset_path

    for file in os.listdir(read_path):
        abs_path = os.path.join(read_path, file)
        content = _read_file(abs_path)
        if ground_truth:
            abs_path = os.path.join(dataset_path, file)
            content = _extract_with_css_selector(content, '[__boilernet_label="1"]')
        source = os.path.splitext(file)[0]
        yield _file_hash(abs_path), _extract_dict('google-trends-2017', source, content, is_truth=ground_truth)


def read_dataset(dataset, ground_truth):
    match dataset:
        case 'cetd':
            return read_cetd(ground_truth)
        case 'cleaneval':
            return read_cleaneval(ground_truth)
        case 'cleanportaleval':
            return read_cleanportaleval(ground_truth)
        case 'dragnet':
            return read_dragnet(ground_truth)
        case 'google-trends-2017':
            return read_google_trends_2017(ground_truth)
        case 'l3s-gn1':
            return read_l3s_gn1(ground_truth)
        case 'readability':
            return read_readability(ground_truth)
        case 'scrapinghub':
            return read_scrapinghub_benchmark(ground_truth)
        case _:
            raise ValueError(f'Invalid dataset: {dataset}')
