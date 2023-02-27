# Web Content Extraction Benchmark

This repository contains code and data for the paper *"An Empirical Comparison of Web Content Extraction Algorithms"* (Bevendorff et al., 2023).

The paper is a reproducibility study of state-of-the-art web page main content extraction tools. This repository provides both the raw data from the paper and the tools used for creating them. Following are usage instructions for combining existing annotated datasets to a common format and for running and evaluating the content extraction systems on this combined dataset.

## Install dependencies

There are two ways you can install the dependencies to run the code.

### Using Poetry (recommended)

If you have the [Poetry](https://python-poetry.org/) package manager for Python installed already, you can simply set up everything with:

```console
poetry install && poetry shell
```
After the installation of all dependencies, you will end up in a new shell with a loaded venv. You can exit the shell at any time with `exit`.

### Using Pip (alternative)

You can also create a venv yourself and use `pip` to install dependencies:

```console
python3 -m venv venv
source venv/bin/activate
pip install .
```

## Extract data
In the next step, you should extract the datasets. There is a compressed tarball called ``datasets/combined.tar.xz``, which contains all eight datasets in a common format. Extract it into the same directory as follows (run this from within the `datasets` directory):

```console
tar xf combined.tar.xz
```

Alternatively, you can also extract the original raw datasets:

```console
tar xf raw.tar.xz
```

and then convert them yourself (the result will be the same as if you ran the step above). After extracting the raw data, run this from the root directory of the repository:

```console
wceb convert-datasets
```

## Run extraction models

You can run extraction models (from the repository root) with

```console
wceb extract
```

After user confirmation, this will run all extraction models on the datasets and place the results under `outputs/model-outputs`. This directory also contains another tarball with the extraction results from the original study, which you can reuse (running the extractors will take a while).

To run only specific models, specify them with the `-m` flag. To include only specific datasets, specify them with the `-d` flag. For instance:

```console
wceb extract -m readability -m resiliparse -d scrapinghub
```

This will run only Readability and Resiliparse on the Scrapinghub dataset. Enter `wceb extract --help` for more information.


### Run Web2Text

By default, Web2Text is excluded from the list, since it is extremely slow and requires a few extra setup steps.

**NOTE:** A working Python 3.7 installation is required on your system.

First, make sure, all Git submodules are checked out:

```console
git submodule update --init --recursive
```

Then switch to the `third-party/web2text` subdirectory and run the following commands:

```console
python3.7 -m venv venv
source venv/bin/activate
pip install numpy==1.18.0 tensorflow==1.15.0 tensorflow-gpu==1.15.0 protobuf==3.20.1 future==0.18.3
deactivate
```

Back from the repository root, you can now run Web2Text:

```console
wceb extract -m web2text -p 1
```

## Evaluate extraction results

To evaluate the extraction results, run

```console
wceb eval score [all|rouge|levenshtein]
```

This will calculate the ROUGE-LSum and Levenshtein scores for the existing extractions.

Be aware that ROUGE-LSum is quite slow. To speed it up a bit, you can cythonize the `rouge-score` module first:

```console
wceb eval cythonize-rouge
```

The calculated scores will be stored in `outputs/metrics-computed`. The `output` directory already contains a zipped version of the original results from the study.

### Aggregate and visualize scores

To aggregate the results, you first need to calculate the page extraction complexities (unless you extracted the provided tarball):

```console
wceb complexity calculate
```

Afterwards, you can aggregate the extraction performance scores:

```console
wceb eval aggregate [all|rouge|levenshtein]
```

This will create aggregated metrics and visualizations under `outputs/metrics-computed`.

You can also visualize the previously computed complexities with

```console
wceb complexity visualize
```

The visualizations will be written to `outputs/metrics-computed/_complexity`.

### Cluster pages

As a rough approximation for page complexity, you can cluster the pages based on simple HTML features and visualize the clusters using the following three commands:

```console
wceb complexity extract-features
wceb complexity cluster
wceb complexity visualize-clusters
```

The visualizations will be written to the same output directory as the complexity scores.

## Cite

The paper can be cited as follows:

```bibtex
@Article{bevendorff:2023,
  author = {Janek Bevendorff and Sanket Gupta and Johannes Kiesel and Benno Stein},
  title  = {{An Empirical Comparison of Web Content Extraction Algorithms}},
  year   = 2023
}
```
