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

import os

ROOT_PATH = os.path.realpath(os.path.join(os.getcwd()))
DATASET_PATH = os.path.join(ROOT_PATH, 'datasets')

DATASET_RAW_PATH = os.path.join(DATASET_PATH, 'raw')
DATASET_COMBINED_PATH = os.path.join(DATASET_PATH, 'combined')
DATASET_COMBINED_TRUTH_PATH = os.path.join(DATASET_COMBINED_PATH, 'ground-truth')
DATASET_COMBINED_HTML_PATH = os.path.join(DATASET_COMBINED_PATH, 'html')

OUTPUTS_PATH = os.path.join(ROOT_PATH, 'outputs')
HTML_FEATURES_PATH = os.path.join(OUTPUTS_PATH, 'html-features')
MODEL_OUTPUTS_PATH = os.path.join(OUTPUTS_PATH, 'model-outputs')
METRICS_PATH = os.path.join(OUTPUTS_PATH, 'metrics-computed')
METRICS_AGG_PATH = os.path.join(METRICS_PATH, '_aggregated')
METRICS_COMPLEXITY_PATH = os.path.join(METRICS_PATH, '_complexity')
