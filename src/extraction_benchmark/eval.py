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
from Levenshtein import ratio as levenshtein_ratio
from multiprocessing import get_context
import re
from itertools import pairwise, product

import click
from rouge_score import rouge_scorer, tokenizers
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from extraction_benchmark.globals import *

_TOKEN_RE = re.compile(r'\s+', flags=re.UNICODE | re.MULTILINE)


class Tokenizer(tokenizers.Tokenizer):
    def tokenize(self, text):
        text = text.strip()
        if not text:
            return []
        return _TOKEN_RE.split(text)


def rouge_eval(key, model, dataset, target, pred):
    rouge = rouge_scorer.RougeScorer(['rougeLsum'], use_stemmer=False, split_summaries=True, tokenizer=Tokenizer())

    scores = []
    score = rouge.score(target, pred)
    for s in score:
        t = dict()
        t['hash_key'] = key
        t['model'] = model
        t['prec'] = score[s].precision
        t['rec'] = score[s].recall
        t['f1'] = score[s].fmeasure
        t['scorer'] = s
        t['dataset'] = dataset

        if target.strip() == '':
            t['rec'] = 1.0
            if pred.strip() == '':
                t['prec'] = 1.0
                t['f1'] = 1.0

        scores.append(t)

    return scores


def levenshtein_eval(key, model, dataset, target, pred):
    tokenizer = Tokenizer()
    target = tokenizer.tokenize(target)
    pred = tokenizer.tokenize(pred)
    return [dict(
        hash_key=key,
        model=model,
        dist=levenshtein_ratio(target, pred),
        scorer='levenshtein',
        dataset=dataset
    )]


def _eval_expand_args(args):
    scorer, model, dataset, answer_path, gt_path = args

    if scorer == 'rouge':
        scorer_func = rouge_eval
    elif scorer == 'levenshtein':
        scorer_func = levenshtein_eval
    else:
        raise ValueError('Illegal scorer')

    try:
        model_answers = json.load(open(answer_path, 'r'))
        ground_truth = json.load(open(gt_path, 'r'))
    except:
        return

    scores = []
    for key in ground_truth.keys():
        target = ground_truth[key].get('articleBody', '') or ''
        pred = model_answers.get(key, {}).get('articleBody', '') or ''
        scores.extend(scorer_func(key, model, dataset, target, pred))

    df = pd.DataFrame(scores)
    df.set_index('hash_key')
    store_path = os.path.join(METRICS_PATH, scorer, dataset)
    os.makedirs(store_path, exist_ok=True)
    df.to_csv(os.path.join(store_path, f'{scorer}_{model}.csv'), index=False)


def calculcate_scores(metrics, datasets, models, parallelism):
    """
    Calculate performance scores for pages against the ground truth.

    :param metrics: list of performance scores to calculate (``"rouge"`` or ``"levenshtein"``)
    :param datasets: list of dataset names
    :param models: list of models to evaluate
    :param parallelism: number of parallel workers to run
    """
    jobs = []
    for ds in tqdm(datasets, desc='Loading extractions', leave=False):
        ground_truth_path = os.path.join(DATASET_COMBINED_TRUTH_PATH, ds, f'{ds}.json')
        if not os.path.isfile(ground_truth_path):
            continue

        for mod in models:
            model_answer_path = os.path.join(MODEL_OUTPUTS_PATH, ds, mod, f'{mod}.json')
            if os.path.isfile(model_answer_path):
                jobs.extend([met, mod, ds, model_answer_path, ground_truth_path] for met in metrics)

    with get_context('spawn').Pool(processes=parallelism) as pool:
        for _ in tqdm(pool.imap_unordered(_eval_expand_args, jobs),
                      total=len(jobs), desc='Evaluating model answers'):
            pass


def _layout_ax(ax, angle_xticks=True, hlines=True):
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    if hlines:
        for y in ax.get_yticks():
            ax.axhline(y, linewidth=0.25, color='lightgrey', zorder=-1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if angle_xticks:
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')


def _map_axis_tick_labels(axis):
    ticklabels = axis.get_ticklabels()

    for t in ticklabels:
        if t.get_text().startswith('ensemble_'):
            t.set_color('steelblue')
        elif t.get_text() in ['bs4', 'html_text', 'inscriptis', 'lxml_cleaner', 'xpath_text']:
            t.set_color('gray')

        t.set_text(MODELS.get(t.get_text(), t.get_text()))

    axis.set_ticks(range(len(ticklabels)))
    axis.set_ticklabels(ticklabels)


def _draw_performance_boxsubplot(ax, model_scores, xlabels, ylabel):
    ax.boxplot(
        model_scores,
        positions=range(len(xlabels)),
        labels=xlabels,
        showfliers=False
    )
    _map_axis_tick_labels(ax.xaxis)
    ax.set_ylabel(ylabel)
    ax.set_ylim((-0.1, 1.1))
    _layout_ax(ax)


def _draw_performance_barsubplot(ax, model_scores, lower_err, upper_err, xlabels, ylabel):
    ax.bar(
        xlabels,
        model_scores,
        yerr=(lower_err, upper_err),
        color='sandybrown',
        error_kw=dict(lw=0.75, capsize=5, capthick=0.75),
    )
    _map_axis_tick_labels(ax.xaxis)
    ax.set_ylabel(ylabel)
    ax.set_ylim((-0.1, 1.1))
    _layout_ax(ax)


def _draw_performance_plot(plot_type, data, layout, suptitle, file_suffix):
    fig, axs = plt.subplots(*layout, figsize=(10, 3.5 * len(data)), dpi=200)

    if layout == (1, 1):
        axs = [axs]
    for ax, d in zip(axs, data):
        if plot_type == 'box':
            _draw_performance_boxsubplot(ax, *d)
        else:
            _draw_performance_barsubplot(ax, *d)

    plt.suptitle(suptitle)
    plt.tight_layout()
    plt.savefig(os.path.join(METRICS_PATH, f'{file_suffix}_{plot_type}.png'))
    plt.savefig(os.path.join(METRICS_PATH, f'{file_suffix}_{plot_type}.pdf'))


def _sort_vectors(*vals, reverse=True):
    """Sort multiple vectors / lists by values in the first one."""
    return zip(*sorted(zip(*vals), key=lambda x: x[0], reverse=reverse))


def aggregate_scores(score_name, models, datasets, complexity):
    """
    Aggregate evaluation statistics.

    :param score_name: score to aggregated (``"rouge"`` or ``"levenshtein"``)
    :param models: list of input model names
    :param datasets: list of input dataset names
    :param complexity: list of complexity classes to include
    """
    score_in_path = os.path.join(METRICS_PATH, score_name)
    if not os.path.isdir(score_in_path):
        return

    if score_name == 'rouge':
        score_cols = ['prec', 'rec', 'f1']
        main_score_col = 'f1'
    else:
        score_cols = ['dist']
        main_score_col = 'dist'

    comp_quant_path = os.path.join(DATASET_COMBINED_TRUTH_PATH, 'complexity_quantiles.csv')
    q = pd.read_csv(comp_quant_path, index_col=0)
    compl_range = {'all': None}
    compl_range.update({k: v for k, v in zip(COMPLEXITIES, pairwise([0, float(q.loc[0.25]), float(q.loc[0.75]), 1]))})

    boxplot_data = []
    barplot_data = []
    for comp in complexity:
        in_df = pd.DataFrame()
        for d, m in tqdm(list(product(datasets, models)), desc=f'Loading score frames (complexity: {comp})'):
            p = os.path.join(score_in_path, d, f'{score_name}_{m}.csv')
            if not os.path.isfile(p):
                continue

            df = pd.read_csv(p, index_col=['model', 'dataset'])
            if compl_range[comp] is not None:
                # Filter input dataframe to include only pages within chosen complexity range
                c = pd.read_csv(os.path.join(DATASET_COMBINED_TRUTH_PATH, d, f'{d}_complexity.csv'), index_col='hash_key')
                c = c[(c['complexity'] >= compl_range[comp][0]) & (c['complexity'] <= compl_range[comp][1])]
                df = df[df['hash_key'].isin(c.index)]

            df.set_index(['hash_key'], append=True, inplace=True)
            in_df = pd.concat([in_df, df])

        models = sorted(in_df.index.unique('model'))

        os.makedirs(METRICS_PATH, exist_ok=True)
        out_df = pd.DataFrame(columns=['model', 'dataset',
                                       *[f'mean_{c}' for c in score_cols],
                                       *[f'median_{c}' for c in score_cols]])
        out_df.set_index(['model', 'dataset'], inplace=True)

        model_f1_scores = []
        model_f1_medians = []
        model_f1_means = []
        model_f1_lower_err = []
        model_f1_upper_err = []
        for i, m in tqdm(enumerate(models), desc='Calculating model stats', leave=False):
            model_df = in_df.loc[m, :, :].drop(columns=['scorer'])

            model_ds_group = model_df.groupby('dataset')
            mean_ds = model_ds_group.mean()
            median_ds = model_ds_group.median()

            ds_stats = pd.concat([mean_ds, median_ds], axis=1)

            mean_micro = model_df.mean()
            median_micro = model_df.median()
            micro = pd.concat([mean_micro, median_micro])
            micro.name = '_micro'
            ds_stats = pd.concat([ds_stats, micro])

            mean_macro = mean_ds.mean()
            median_macro = median_ds.median()
            macro = pd.concat([mean_macro, median_macro])
            macro.name = '_macro'
            ds_stats = pd.concat([ds_stats, macro])

            ds_stats.columns = out_df.columns

            ds_stats['model'] = m
            ds_stats = ds_stats.reset_index().set_index(['model', 'dataset'])

            out_df = pd.concat([out_df, ds_stats.round(3)])

            model_f1_scores.append(model_df[main_score_col])
            model_f1_medians.append(median_micro[main_score_col])

            model_f1_means.append(mean_micro[main_score_col])
            model_f1_lower_err.append(abs(mean_micro[main_score_col] - model_df[main_score_col].quantile(0.25)))
            model_f1_upper_err.append(abs(model_df[main_score_col].quantile(0.75) - mean_micro[main_score_col]))

        _, f1, labels = _sort_vectors(model_f1_medians, model_f1_scores, models)
        boxplot_data.append([f1, labels, f'Complexity: {comp.capitalize()}'])

        f1, low, high, labels = _sort_vectors(model_f1_means, model_f1_lower_err, model_f1_upper_err, models)
        barplot_data.append([f1, low, high, labels, f'Complexity: {comp.capitalize()}'])

        file_suffix = score_name
        if comp != 'all':
            file_suffix += f'_complexity_{comp}'

        out_df_max = out_df.groupby(['dataset']).max()

        def _highlight_max_per_ds(s):
            def _is_max(idx, val):
                return val >= out_df_max[s.name][idx[1]]
            return ['font-weight: bold' if _is_max(idx, val) else '' for idx, val in s.items()]

        # Remap series to friendly names
        def _remap_series_names(s):
            s.index = pd.Index(data=[MODELS.get(n, n) for n in s.index.values], name='Model')
            return s

        out_styler = out_df.style.apply(_highlight_max_per_ds).format(precision=3)
        out_styler.to_excel(os.path.join(METRICS_PATH, f'{file_suffix}.xlsx'))

        # Compile reduced versions of the table with global averages
        for series in '_micro', '_macro':
            out_df_reduced = out_df.loc[:, series, :].droplevel('dataset').sort_values(
                f'mean_{main_score_col}', ascending=False)
            out_df_reduced.name = 'Model'
            if score_name == 'rouge':
                out_df_reduced.columns = ['Mean Precision', 'Mean Recall', 'Mean F1',
                                          'Median Precision', 'Median Recall', 'Median F1']
            else:
                out_df_reduced.columns = ['Mean Distance', 'Median Distance']
            out_df_reduced = out_df_reduced.apply(_remap_series_names)

            # XLSX
            out_styler = out_df_reduced.style.highlight_max(props='font-weight: bold').format(precision=3)
            out_styler.to_excel(os.path.join(METRICS_PATH, f'{file_suffix}{series}.xlsx'), float_format='%.3f')

            # LaTeX
            if score_name == 'rouge':
                out_df_reduced.columns = ['Mean Precision', 'Mean Recall', 'Mean $F_1$',
                                          'Median Precision', 'Median Recall', 'Median $F_1$']
            out_styler = out_df_reduced.style.highlight_max(props=r'bf:').format(precision=3)
            out_styler.to_latex(os.path.join(METRICS_PATH, f'{file_suffix}{series}.tex'))

    if score_name == 'rouge':
        title_box = 'ROUGE-LSum Median $F_1$ Page Scores'
        title_bar = 'ROUGE-LSum Mean $F_1$ Page Scores (Macro Average)'
    else:
        title_box = 'Normalized Median Levenshtein Distances'
        title_bar = 'Normalized Mean Levenshtein Distance (Macro Average)'

    _draw_performance_plot(
        'box',
        boxplot_data,
        (len(boxplot_data), 1),
        title_box,
        score_name)

    _draw_performance_plot(
        'bar',
        barplot_data,
        (len(barplot_data), 1),
        title_bar,
        score_name)

    click.echo(f'Aggregation written to "{METRICS_PATH}"')
