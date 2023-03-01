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

import json
import re


def read_jsonl(file):
    """
    Read JSONL file and return iterable of dicts.

    :param file: input filename
    :return: iterable of dicts
    """
    with open(file, 'r') as f:
        for line in f:
            yield json.loads(line)


def jsonl_to_dict(file):
    """
    Load a JSONL into a single dict with ``"page_id"`` as keys.

    :param file: input file name
    :return: assembled dict
    """
    loaded = {}
    for j in read_jsonl(file):
        loaded[j['page_id']] = {k: v for k, v in j.items() if k != 'page_id'}
    return loaded


_TOKEN_RE_WS = re.compile(r'\s+', flags=re.UNICODE | re.MULTILINE)


def tokenize_ws(text):
    """
    Tokenize text by white space.

    :param text: input text
    :return: list of tokens
    """
    text = text.strip()
    if not text:
        return []
    return _TOKEN_RE_WS.split(text)


_TOKEN_RE_WORDS = re.compile(r'\w+', flags=re.UNICODE)


def tokenize_words(text):
    """
    Tokenize text by extracting Unicode word tokens (skips any non-word tokens).

    :param text: input text
    :return: list of tokens
    """
    return _TOKEN_RE_WORDS.findall(text)
