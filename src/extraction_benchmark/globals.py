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

from extraction_benchmark.paths import *
from extraction_benchmark.extractors import list_extractors

_DATASET_FRIENDLY_NAME_MAP = {
    'cetd': 'CETD',
    'cleaneval': 'CleanEval',
    'cleanportaleval': 'CleanPortalEval',
    'dragnet': 'Dragnet',
    'google-trends-2017': 'Google-Trends',
    'l3s-gn1': 'L3S-GN1',
    'readability': 'Readability',
    'scrapinghub': 'Scrapinghub'
}

if os.path.isdir(DATASET_RAW_PATH):
    DATASETS = {k: _DATASET_FRIENDLY_NAME_MAP.get(k, k) for k in os.listdir(DATASET_RAW_PATH)
                if os.path.isdir(os.path.join(DATASET_RAW_PATH, k))}
else:
    DATASETS = {}

_MODEL_FRIENDLY_NAME_MAP = dict(
    ensemble_best='(Best only)',
    ensemble_weighted='(Best weighted)',
    ensemble_majority='(Majority all)',

    bs4='BS4',
    boilernet='BoilerNet',
    boilerpipe='Boilerpipe',
    bte='BTE',
    dragnet='Dragnet',
    extractnet='ExtractNet',
    go_domdistiller='DOM Distiller',
    goose3='Goose3',
    justext='jusText',
    lxml_cleaner='lxml Cleaner',
    news_please='news-please',
    newspaper3k='Newspaper3k',
    readability='Readability',
    resiliparse='Resiliparse',
    trafilatura='Trafilatura',
    web2text='Web2Text',
    xpath_text='XPath Text',
)

MODELS = {k: _MODEL_FRIENDLY_NAME_MAP.get(k, k)
          for k in list_extractors(names_only=True, include_ensembles=False)}
MODELS_ALL = {k: _MODEL_FRIENDLY_NAME_MAP.get(k, k)
              for k in list_extractors(names_only=True, include_ensembles=True)}
MODELS_ENSEMBLE = [m for m in MODELS_ALL if m.startswith('ensemble_')]
MODELS_BASELINE = ['bs4', 'html_text', 'inscriptis', 'lxml_cleaner', 'xpath_text']

SCORES = [
    'rouge',
    'levenshtein'
]

COMPLEXITIES = [
    'low',
    'medium',
    'high'
]
