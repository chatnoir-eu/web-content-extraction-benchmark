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

import click
from extraction_benchmark.cli import *


@click.group()
def main():
    """
    Web Content Extraction Benchmark.

    Reproduction study of various main content extraction / boilerplate removal tools from
    the scientific literature and the open source community.
    """
    pass


main.add_command(complexity)
main.add_command(eval)
main.add_command(extract)
main.add_command(convert_datasets)


if __name__ == '__main__':
    main()
