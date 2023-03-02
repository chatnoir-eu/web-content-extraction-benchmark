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

from itertools import pairwise, product
import math
from multiprocessing import get_context
import os

import click
from Levenshtein import ratio as levenshtein_ratio
import pandas as pd
from rouge_score import rouge_scorer, tokenizers
from tqdm import tqdm

from extraction_benchmark.globals import *
from extraction_benchmark import plt
from extraction_benchmark.util import jsonl_to_dict, read_jsonl, tokenize_ws


class Tokenizer(tokenizers.Tokenizer):
    def tokenize(self, text):
        return tokenize_ws(text)


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

    ground_truth = jsonl_to_dict(gt_path)
    df = pd.DataFrame()
    for model_answer in read_jsonl(answer_path):
        if model_answer['page_id'] not in ground_truth:
            continue
        target = ground_truth[model_answer['page_id']].get('plaintext') or ''
        pred = model_answer.get('plaintext') or ''
        df = pd.concat([df, pd.DataFrame(scorer_func(model_answer['page_id'], model, dataset, target, pred))])

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
        ground_truth_path = os.path.join(DATASET_COMBINED_TRUTH_PATH, f'{ds}.jsonl')
        if not os.path.isfile(ground_truth_path):
            continue

        for model in models:
            model_answer_path = os.path.join(MODEL_OUTPUTS_PATH, ds,  f'{model}.jsonl')
            if os.path.isfile(model_answer_path):
                jobs.extend([met, model, ds, model_answer_path, ground_truth_path] for met in metrics)

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


def _map_model_label(label):
    if label.get_text() in MODELS_ENSEMBLE:
        label.set_color('#1767b0')
    elif label.get_text() in MODELS_BASELINE:
        label.set_color('gray')
    label.set_text(MODELS_ALL.get(label.get_text(), label.get_text()))
    return label


def _map_axis_tick_labels(axis):
    ticklabels = [_map_model_label(t) for t in axis.get_ticklabels()]
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
        width=0.7,
        yerr=(lower_err, upper_err),
        error_kw=dict(lw=0.75, capthick=0.75, ecolor=plt.ERROR_BAR_COLOR),
    )
    _map_axis_tick_labels(ax.xaxis)
    ax.set_ylabel(ylabel)
    ax.set_ylim((0.0, 1.1))
    ax.set_xlim((-0.7, len(xlabels) - 0.3))
    _layout_ax(ax)


def _draw_performance_plot(plot_type, data, layout, suptitle, score_name):
    fig, axs = plt.subplots(*layout, figsize=(9.5, 3.5 * len(data)))

    if layout == (1, 1):
        axs = [axs]
    for ax, d in zip(axs, data):
        if plot_type == 'box':
            _draw_performance_boxsubplot(ax, *d)
        else:
            _draw_performance_barsubplot(ax, *d)

    plt.suptitle(suptitle)
    plt.tight_layout()
    plt.savefig(os.path.join(METRICS_PATH, score_name, f'{score_name}_{plot_type}.pdf'))
    plt.close()


def _sort_vectors(*vals, reverse=True):
    """Sort multiple vectors / lists by values in the first one."""
    return zip(*sorted(zip(*vals), key=lambda x: x[0], reverse=reverse))


def _write_agg_df_to_files(df, score_name, main_score_col, out_file_base):
    out_df_max = df.groupby(['dataset']).max()

    def _highlight_max_per_ds(s):
        def _is_max(idx, val):
            return val >= out_df_max[s.name][idx[1]]

        return ['font-weight: bold' if _is_max(idx, val) else '' for idx, val in s.items()]

    # Remap series to friendly names
    def _remap_series_names(s):
        s.index = pd.Index(data=[MODELS_ALL.get(n, n) for n in s.index.values], name='Model')
        return s

    os.makedirs(os.path.join(METRICS_PATH, score_name), exist_ok=True)
    out_styler = df.style.apply(_highlight_max_per_ds).format(precision=3)
    out_styler.to_excel(out_file_base + '.xlsx')

    # Compile reduced versions of the table with global averages
    for series in '_micro', '_macro':
        out_df_reduced = df.loc[:, series, :].sort_values(f'mean_{main_score_col}', ascending=False)
        out_df_reduced.name = 'Model'
        if score_name == 'rouge':
            out_df_reduced.columns = ['Mean Precision', 'Mean Recall', 'Mean F1',
                                      'Median Precision', 'Median Recall', 'Median F1']
        else:
            out_df_reduced.columns = ['Mean Distance', 'Median Distance']
        out_df_reduced = out_df_reduced.apply(_remap_series_names)

        # XLSX
        out_styler = out_df_reduced.style.highlight_max(props='font-weight: bold').format(precision=3)
        out_styler.to_excel(out_file_base + f'{series}.xlsx', float_format='%.3f')

        # LaTeX
        if score_name == 'rouge':
            out_df_reduced.columns = ['Mean Precision', 'Mean Recall', 'Mean $F_1$',
                                      'Median Precision', 'Median Recall', 'Median $F_1$']
        out_styler = out_df_reduced.style.highlight_max(props=r'bf:').format(precision=3)
        out_styler.to_latex(os.path.join(out_file_base + f'{series}.tex'))


def _agg_model_at_complexity(complexity, in_df, score_name, score_cols, main_score_col):
    models = sorted(in_df.index.unique('model'))

    out_df = pd.DataFrame(columns=['model', 'dataset',
                                   *[f'mean_{c}' for c in score_cols],
                                   *[f'median_{c}' for c in score_cols]])
    out_df.set_index(['model', 'dataset'], inplace=True)

    model_main_scores = []
    model_main_medians = []
    model_main_means = []
    model_main_lower_err = []
    model_main_upper_err = []
    for m in models:
        model_df = in_df.loc[m, :, :].drop(columns=['scorer'])

        model_ds_group = model_df.groupby('dataset')
        mean_ds = model_ds_group.mean()
        median_ds = model_ds_group.median()

        ds_stats = pd.concat([mean_ds, median_ds], axis=1)

        mean_micro = model_df.mean()
        median_micro = model_df.median()
        micro = pd.concat([mean_micro, median_micro]).to_frame('_micro').T
        micro.index.name = 'dataset'
        ds_stats = pd.concat([ds_stats, micro])

        mean_macro = mean_ds.mean()
        median_macro = median_ds.median()
        macro = pd.concat([mean_macro, median_macro]).to_frame('_macro').T
        macro.index.name = 'dataset'
        ds_stats = pd.concat([ds_stats, macro])

        ds_stats.columns = out_df.columns

        ds_stats['model'] = m
        ds_stats = ds_stats.reset_index().set_index(['model', 'dataset'])

        out_df = pd.concat([out_df, ds_stats.round(3)])

        model_main_scores.append(model_df[main_score_col])
        model_main_medians.append(median_micro[main_score_col])

        model_main_means.append(mean_micro[main_score_col])
        model_main_lower_err.append(abs(mean_micro[main_score_col] - model_df[main_score_col].quantile(0.25)))
        model_main_upper_err.append(abs(model_df[main_score_col].quantile(0.75) - mean_micro[main_score_col]))

    file_suffix = f'_complexity_{complexity}' if complexity != 'all' else ''
    _write_agg_df_to_files(out_df, score_name, main_score_col,
                           os.path.join(METRICS_PATH, score_name, f'{score_name}{file_suffix}'))

    _, main_scores, labels = _sort_vectors(model_main_medians, model_main_scores, models)
    boxplot_data = [main_scores, labels, f'Complexity: {complexity.capitalize()}']

    main_scores, low, high, labels = _sort_vectors(model_main_means, model_main_lower_err, model_main_upper_err, models)
    barplot_data = [main_scores, low, high, labels, f'Complexity: {complexity.capitalize()}']

    return boxplot_data, barplot_data


def _plot_score_histograms(title, score_df, out_file):
    models = sorted(score_df.index.unique('model'), key=lambda m: score_df[m, :, :].median(), reverse=True)
    cols = 4
    rows = math.ceil(len(models) / cols)

    fig, axs = plt.subplots(rows, cols, sharey=True, figsize=(2 * cols, 2 * rows))
    for ax, m in zip(axs.flatten(), models):
        ax.hist(
            score_df[m, :, :],
            bins=25
        )
        ax.axvline(score_df[m, :, :].median(), color=plt.MEDIAN_BAR_COLOR, linewidth=1)
        ax.set_ylabel(m)
        _map_model_label(ax.yaxis.get_label())
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticklabels([])

    # Hide empty plots
    if len(models) % cols:
        [ax.set_visible(False) for ax in axs[-1][len(models) % cols:].flatten()]

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()


def aggregate_scores(score_name, models, datasets, complexities):
    """
    Aggregate evaluation statistics.

    :param score_name: score to aggregated (``"rouge"`` or ``"levenshtein"``)
    :param models: list of input model names
    :param datasets: list of input dataset names
    :param complexities: list of complexity classes to include
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

    comp_quant_path = os.path.join(METRICS_COMPLEXITY_PATH, 'complexity_quantiles.csv')
    q = pd.read_csv(comp_quant_path, index_col=0)
    compl_range = {'all': None}
    compl_range.update({k: v for k, v in zip(COMPLEXITIES, pairwise([0, float(q.loc[0.25]), float(q.loc[0.75]), 1]))})

    if score_name == 'rouge':
        title_box = 'ROUGE-LSum Median $F_1$ Page Scores'
        title_bar = 'ROUGE-LSum Mean $F_1$ Page Scores (Macro Average)'
        title_hist = 'ROUGE-LSum $F_1$ Page Scores'
    else:
        title_hist = 'Normalized Levenshtein Distances'
        title_box = 'Normalized Median Levenshtein Distances'
        title_bar = 'Normalized Mean Levenshtein Distance (Macro Average)'

    with click.progressbar(complexities, label=f'Aggregating "{score_name}" scores') as progress:
        boxplot_data = []
        barplot_data = []
        for comp in progress:
            score_df = pd.DataFrame()
            for d, m in product(datasets, models):
                p = os.path.join(score_in_path, d, f'{score_name}_{m}.csv')
                if not os.path.isfile(p):
                    continue

                df = pd.read_csv(p, index_col=['model', 'dataset'])
                if compl_range[comp] is not None:
                    # Filter input dataframe to include only pages within chosen complexity range
                    c = pd.read_csv(os.path.join(METRICS_COMPLEXITY_PATH, d, f'{d}_complexity.csv'),
                                    index_col='hash_key')
                    c = c[(c['complexity'] >= compl_range[comp][0]) & (c['complexity'] <= compl_range[comp][1])]
                    df = df[df['hash_key'].isin(c.index)]

                df.set_index(['hash_key'], append=True, inplace=True)
                score_df = pd.concat([score_df, df])

            hist_file_prefix = f'_complexity_{comp}' if comp != 'all' else ''
            _plot_score_histograms(f'{title_hist} (Complexity: {comp.capitalize()})', score_df[main_score_col],
                                   os.path.join(score_in_path, f'{score_name}{hist_file_prefix}_hist.pdf'))

            box, bar = _agg_model_at_complexity(comp, score_df, score_name, score_cols, main_score_col)
            boxplot_data.append(box)
            barplot_data.append(bar)

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
