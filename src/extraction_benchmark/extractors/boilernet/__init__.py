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

from collections import defaultdict
import json
import os
import warnings

import numpy as np
from bs4 import BeautifulSoup
import nltk

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from .net.preprocess import get_feature_vector, get_leaves, process


BOILERNET_ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

_model = None
_word_map = None
_tag_map = None


def load_model():
    global _model, _word_map, _tag_map
    if not _model:
        _model = tf.keras.models.load_model(os.path.join(BOILERNET_ROOT_PATH, 'model.h5'))
        nltk.download('punkt', quiet=True)
        with open(os.path.join(BOILERNET_ROOT_PATH, 'words.json')) as f:
            _word_map = json.load(f)
        with open(os.path.join(BOILERNET_ROOT_PATH, 'tags.json')) as f:
            _tag_map = json.load(f)
    return _model, _word_map, _tag_map


def extract(html):
    model, word_map, tag_map = load_model()

    tags = defaultdict(int)
    words = defaultdict(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        doc = BeautifulSoup(html, features='html5lib')
    processed = process(doc, tags, words)
    if not processed:
        return ''

    inputs = [get_feature_vector(w, t, word_map, tag_map) for w, t, _ in processed]
    inputs = np.expand_dims(np.stack(inputs), 0)
    predicted = np.around(model.predict(inputs, verbose=0))

    main_content = ''
    doc = BeautifulSoup(html, features='html5lib')
    for i, (leaf, _, _) in enumerate(get_leaves(doc.find_all('html')[0])):
        if predicted[0, i, 0]:
            main_content += leaf + '\n'
    return main_content.strip()
