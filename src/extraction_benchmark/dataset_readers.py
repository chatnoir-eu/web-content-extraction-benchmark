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

from abc import ABC, abstractmethod
import glob
import gzip
import hashlib
import json
import re
from typing import Any, Dict, Iterable, Optional, Tuple

from resiliparse.parse import bytes_to_str, detect_encoding
from resiliparse.parse.html import HTMLTree, NodeType

from extraction_benchmark.paths import *


class DatasetReader(ABC):
    """Abstract dataset reader class."""

    def __init__(self, ground_truth):
        """
        Initialize dataset reader.

        :param ground_truth: whether the reader should return the raw HTML data or the ground truth.
        """
        self.is_truth = ground_truth
        self._iter = iter(self.read())

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._iter)

    def __len__(self):
        return self.dataset_size()

    @abstractmethod
    def read(self) -> Iterable[Tuple[str, Dict[str, Any]]]:
        """
        Return an iterable over the items in the dataset.

        Returned items should be tuples with the case / page ID and the case / page data as a dict.
        The dicts should contain at least an ``"html"`` or ``"plaintext"`` key, depending on whether
        the iterated dataset is a raw HTML or a ground truth page.
        """
        pass

    @abstractmethod
    def dataset_size(self) -> Optional[int]:
        """
        Return size of dataset or ``None`` if size is unknown.

        :return: size of dataset
        """
        pass

    @staticmethod
    def _hash(data: bytes):
        """
        Return SHA-256 hash of input bytes, which can be used as a an ID.

        :param data: input bytes
        :return: hash of bytes as hex string
        """
        m = hashlib.sha256()
        m.update(data)
        return m.hexdigest()

    @classmethod
    def _file_hash(cls, file: str):
        """
        Return SHA-256 hash of a file that can be used as a page ID.

        :param file: input file name
        :return: hash of the file as hex string
        """
        with open(file, 'rb') as f:
            return cls._hash(f.read())

    def _build_dict(self, source_dataset, source_case, content, **kwargs) -> Dict[str, Any]:
        """
        Helper method for creating a dict to return in :meth:`read`.

        :param source_dataset: source dataset name
        :param source_case: source case / file name
        :param content: HTML or plaintext content
        :param kwargs: other key / value pairs to include in the dict
        :return: dict with requested data
        """
        d = {
            ('plaintext' if self.is_truth else 'html'): content,
            **{k: v for k, v in kwargs.items() if v},
            'source': [source_dataset, source_case] if source_case else [source_dataset]
        }
        return d

    @staticmethod
    def _read_file(path, fixed_encoding=None):
        """
        Helper method for reading a file, detecting its encoding and returning the contents as UTF-8 string.
        If the input file is GZip-compressed, it will be decompressed automatically.

        :param path: file path
        :param fixed_encoding: use this fixed encoding instead of trying to detect if from the file
        :return: UTF-8 string of file contents
        """
        with open(path, 'rb') as f:
            file_bytes = f.read()
            if path.endswith('.gz'):
                file_bytes = gzip.decompress(file_bytes)
            if fixed_encoding:
                enc = fixed_encoding
            else:
                enc = detect_encoding(file_bytes, max_len=100000, html5_compatible=False) or 'utf-8'
            return bytes_to_str(file_bytes, encoding=enc, fallback_encodings=['utf-8', 'cp1252'])


class CleanEvalReader(DatasetReader):
    def __init__(self, ground_truth):
        super().__init__(ground_truth)

        self.dataset_name = 'cleaneval'
        self.dataset_path = os.path.join(DATASET_RAW_PATH, self.dataset_name, 'orig')
        self.dataset_path_truth = os.path.join(DATASET_RAW_PATH, self.dataset_name, 'clean')

    def read(self) -> Iterable[Tuple[str, Dict[str, Any]]]:
        read_path = self.dataset_path_truth if self.is_truth else self.dataset_path

        text_tag_re = re.compile(r'(?:^<text [^>]+>\s*|\s*</text>$)', flags=re.MULTILINE)

        for file in os.listdir(read_path):
            abs_path = os.path.join(read_path, file)
            content = self._read_file(abs_path)
            url = None
            if self.is_truth:
                url = re.search(r'^\s*URL: (https?://.+)', content)
                if url:
                    url = url.group(1)
                content = HTMLTree.parse(content).body.text
                content = re.sub(r'\n +', '\n', content)
                content = re.sub(r'^\s*URL:[^\n]+\s*', '', content)    # Strip URL line

            if self.is_truth:
                abs_path = os.path.join(self.dataset_path, os.path.splitext(file)[0] + '.html')
            else:
                content = text_tag_re.sub('', content)
            source = os.path.splitext(file)[0]
            yield self._file_hash(abs_path), self._build_dict(self.dataset_name, source, content, url=url)

    def dataset_size(self) -> Optional[int]:
        return len(glob.glob(os.path.join(DATASET_RAW_PATH, self.dataset_name, 'clean', '*.txt')))


class CleanPortalEvalReader(CleanEvalReader):
    def __init__(self, ground_truth):
        super().__init__(ground_truth)
        self.dataset_name = 'cleanportaleval'
        self.dataset_path = os.path.join(DATASET_RAW_PATH, self.dataset_name, 'input')
        self.dataset_path_truth = os.path.join(DATASET_RAW_PATH, self.dataset_name, 'GoldStandard')

    def dataset_size(self) -> Optional[int]:
        return len(glob.glob(os.path.join(DATASET_RAW_PATH, self.dataset_name, 'GoldStandard', '*.txt')))


class DragnetReader(DatasetReader):
    def read(self) -> Iterable[Tuple[str, Dict[str, Any]]]:
        dataset_path = os.path.join(DATASET_RAW_PATH, 'dragnet', 'HTML')
        dataset_path_truth = os.path.join(DATASET_RAW_PATH, 'dragnet', 'corrected', 'Corrected')
        read_path = dataset_path_truth if self.is_truth else dataset_path

        for file in os.listdir(read_path):
            abs_path = os.path.join(read_path, file)
            content = self._read_file(abs_path)
            if self.is_truth:
                file = os.path.splitext(os.path.splitext(file)[0])[0]
                abs_path = os.path.join(dataset_path, file)
            source = os.path.splitext(file)[0]
            yield self._file_hash(abs_path), self._build_dict('dragnet', source, content)

    def dataset_size(self) -> Optional[int]:
        return len(glob.glob(os.path.join(DATASET_RAW_PATH, 'dragnet', 'corrected', 'Corrected', '*.txt')))


class CETDReader(DatasetReader):
    def __init__(self, ground_truth):
        super().__init__(ground_truth)
        self.verticals = ['arstechnica', 'BBC', 'Chaos', 'nytimes', 'wiki', 'YAHOO!']

    def read(self) -> Iterable[Tuple[str, Dict[str, Any]]]:
        dataset_path = os.path.join(DATASET_RAW_PATH, 'cetd')
        for vertical in self.verticals:
            sub_path = os.path.join(dataset_path, vertical, 'gold' if self.is_truth else 'original')
            for file in os.listdir(sub_path):
                abs_path = os.path.join(sub_path, file)
                content = self._read_file(abs_path)
                if self.is_truth:
                    abs_path = os.path.join(dataset_path, vertical, 'original', os.path.splitext(file)[0] + '.htm')
                source = vertical + '_' + os.path.splitext(file)[0]
                yield self._file_hash(abs_path), self._build_dict('cetd', source, content)

    def dataset_size(self) -> Optional[int]:
        return sum([len(glob.glob(os.path.join(DATASET_RAW_PATH, 'cetd', v, 'gold', '*.txt')))
                    for v in self.verticals])


class ReadabilityReader(DatasetReader):
    def read(self) -> Iterable[Tuple[str, Dict[str, Any]]]:
        dataset_path = os.path.join(DATASET_RAW_PATH, 'readability', 'test-pages')
        for case_dir in os.listdir(dataset_path):
            sub_path = 'expected.html' if self.is_truth else 'source.html'
            abs_path = os.path.join(dataset_path, os.path.join(case_dir, sub_path))
            content = self._read_file(abs_path)
            if self.is_truth:
                content = HTMLTree.parse(content).body.text
                abs_path = os.path.join(dataset_path, os.path.join(case_dir, 'source.html'))
            yield self._file_hash(abs_path), self._build_dict('readability', case_dir, content)

    def dataset_size(self) -> Optional[int]:
        return len(glob.glob(os.path.join(DATASET_RAW_PATH, 'readability', 'test-pages', '*', 'expected.html')))


class ScrapingHubReader(DatasetReader):
    def read(self) -> Iterable[Tuple[str, Dict[str, Any]]]:
        dataset_path = os.path.join(DATASET_RAW_PATH, 'scrapinghub')

        if self.is_truth:
            truth_json = json.load(open(os.path.join(dataset_path, 'ground-truth.json'), 'r'))
            for k, v in truth_json.items():
                yield k, self._build_dict('scrapinghub', None, v['articleBody'], url=v['url'])
            return

        dataset_path = os.path.join(dataset_path, 'html')
        for file in os.listdir(dataset_path):
            abs_path = os.path.join(dataset_path, file)
            hash_id = os.path.splitext(os.path.splitext(file)[0])[0]
            with gzip.GzipFile(abs_path, 'r') as f:
                file_hash = self._hash(f.read())
            yield file_hash, self._build_dict('scrapinghub', hash_id, self._read_file(abs_path))

    def dataset_size(self) -> Optional[int]:
        return len(json.load(open(os.path.join(DATASET_RAW_PATH, 'scrapinghub', 'ground-truth.json'), 'r')))


class L3SGN1Reader(DatasetReader):
    def read(self) -> Iterable[Tuple[str, Dict[str, Any]]]:
        dataset_path = os.path.join(DATASET_RAW_PATH, 'l3s-gn1', 'original')
        dataset_path_truth = os.path.join(DATASET_RAW_PATH, 'l3s-gn1', 'annotated')
        read_path = dataset_path_truth if self.is_truth else dataset_path

        for file in os.listdir(read_path):
            abs_path = os.path.join(read_path, file)
            content = self._read_file(abs_path)
            if self.is_truth:
                abs_path = os.path.join(dataset_path, file)
                content = self._extract_with_css_selector(content, '.x-nc-sel1, .x-nc-sel2, .x-nc-sel3')
            source = os.path.splitext(file)[0]
            yield self._file_hash(abs_path), self._build_dict('l3s-gn1', source, content)

    def dataset_size(self) -> Optional[int]:
        return len(glob.glob(os.path.join(DATASET_RAW_PATH, 'l3s-gn1', 'annotated', '*.html')))

    @staticmethod
    def _extract_with_css_selector(html, selector):
        tree = HTMLTree.parse(html)
        elements = tree.body.query_selector_all(selector)
        content = ''
        for e in elements:
            if len(e.child_nodes) != 1 or e.first_child.type != NodeType.TEXT:
                # Only count leaf nodes to avoid adding an element multiple times
                continue
            if e.parent.tag in ['address', 'article', 'aside', 'blockquote', 'canvas', 'dd', 'div', 'dl', 'dt',
                                'fieldset',
                                'figcaption', 'figure', 'footer', 'form', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'header',
                                'hr', 'li', 'main', 'nav', 'noscript', 'ol', 'p', 'pre', 'section', 'table', 'tfoot',
                                'ul', 'video']:
                content += '\n'
            content += e.text.strip() + ' '
        return content.strip()


class GoogleTrends2017Reader(L3SGN1Reader):
    def read(self) -> Iterable[Tuple[str, Dict[str, Any]]]:
        dataset_path = os.path.join(DATASET_RAW_PATH, 'google-trends-2017', 'raw_html')
        dataset_path_truth = os.path.join(DATASET_RAW_PATH, 'google-trends-2017', 'prepared_html')
        read_path = dataset_path_truth if self.is_truth else dataset_path

        for file in os.listdir(read_path):
            abs_path = os.path.join(read_path, file)
            content = self._read_file(abs_path)
            if self.is_truth:
                abs_path = os.path.join(dataset_path, file)
                content = self._extract_with_css_selector(content, '[__boilernet_label="1"]')
            source = os.path.splitext(file)[0]
            yield self._file_hash(abs_path), self._build_dict('google-trends-2017', source, content)

    def dataset_size(self) -> Optional[int]:
        return len(glob.glob(os.path.join(DATASET_RAW_PATH, 'google-trends-2017', 'prepared_html', '*.html')))


def read_dataset(dataset, ground_truth):
    match dataset:
        case 'cetd':
            return CETDReader(ground_truth)
        case 'cleaneval':
            return CleanEvalReader(ground_truth)
        case 'cleanportaleval':
            return CleanPortalEvalReader(ground_truth)
        case 'dragnet':
            return DragnetReader(ground_truth)
        case 'google-trends-2017':
            return GoogleTrends2017Reader(ground_truth)
        case 'l3s-gn1':
            return L3SGN1Reader(ground_truth)
        case 'readability':
            return ReadabilityReader(ground_truth)
        case 'scrapinghub':
            return ScrapingHubReader(ground_truth)
        case _:
            raise ValueError(f'Invalid dataset: {dataset}')
