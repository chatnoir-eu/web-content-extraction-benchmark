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


from matplotlib.pyplot import *

rcParams['figure.dpi'] = 200

rcParams['pdf.fonttype'] = 42
rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
rcParams['font.family'] = 'sans-serif'

MEDIAN_BAR_COLOR = '#e68a38'
ERROR_BAR_COLOR = '#4d4d4d'

rcParams['errorbar.capsize'] = 4
rcParams['boxplot.meanprops.color'] = 'pink'
rcParams['boxplot.flierprops.marker'] = '.'

# Lighter version of tab10
rcParams['axes.prop_cycle'] = cycler(color=['#53a1d4', '#ff993e', '#56b356'])
