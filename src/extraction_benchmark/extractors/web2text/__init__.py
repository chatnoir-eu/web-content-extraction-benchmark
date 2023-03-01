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

import errno
import hashlib
import os
import subprocess
import tempfile

from extraction_benchmark.paths import THIRD_PARTY_PATH

WEB2TEXT_BASEPATH = os.path.join(THIRD_PARTY_PATH, 'web2text')
WEB2TEXT_PYTHONPATH = os.path.join(WEB2TEXT_BASEPATH, 'src', 'main', 'python')
WEB2TEXT_VENV = os.path.join(WEB2TEXT_BASEPATH, 'venv')


if not os.path.isdir(WEB2TEXT_PYTHONPATH):
    raise FileNotFoundError(errno.ENOENT, WEB2TEXT_BASEPATH,
                            'Could not find Web2Text under current working directory. Please ensure you have the '
                            'submodule checked out and are running this from the repository\'s root directory.')

if not os.path.isdir(WEB2TEXT_VENV):
    raise FileNotFoundError(errno.ENOENT, WEB2TEXT_VENV,
                            'Could not find venv in Web2Text directory. '
                            'Please follow README instructions to create one')


def extract(html):
    scala_cmd = ['scala', '-cp', os.path.join(THIRD_PARTY_PATH, 'web2text.jar')]
    python_cmd = ['python', os.path.join(WEB2TEXT_PYTHONPATH, 'main.py')]
    hash_id = hashlib.sha256(html.encode()).hexdigest()

    proc_env = os.environ.copy()
    proc_env['VIRTUAL_ENV'] = WEB2TEXT_VENV
    proc_env['PATH'] = '{}/bin:{}'.format(proc_env['VIRTUAL_ENV'], proc_env['PATH'])
    proc_env['JAVA_HOME'] = '/usr/lib/jvm/java-8-openjdk-amd64'

    with tempfile.TemporaryDirectory() as tmp_dir:
        file_base = os.path.join(tmp_dir, hash_id)
        html_file = file_base + '.html'
        features_file = file_base + '.features'
        labels_file = file_base + '.labels'
        text_file = file_base + '.txt'

        open(html_file, 'w').write(html)
        exit_code = subprocess.Popen(
            scala_cmd + ['ch.ethz.dalab.web2text.ExtractPageFeatures', html_file, features_file],
            env=proc_env,
            stderr=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL
        ).wait()
        if exit_code != 0:
            raise RuntimeError('Web2Text ExtractPageFeatures failed.')

        exit_code = subprocess.Popen(
            python_cmd + ['classify', features_file, labels_file],
            env=proc_env,
            stderr=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL
        ).wait()
        if exit_code != 0:
            raise RuntimeError('Web2Text DOM node classification failed.')

        exit_code = subprocess.Popen(
            scala_cmd + ['ch.ethz.dalab.web2text.ApplyLabelsToPage', html_file, labels_file, text_file],
            env=proc_env,
            stderr=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL
        ).wait()
        if exit_code != 0:
            raise RuntimeError('Web2Text ApplyLabelsToPage failed.')

        return open(text_file, 'r').read()
