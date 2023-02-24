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

import re


def extract_bs4(html, **_):
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, 'html.parser')
    for e in soup(['script', 'style', 'noscript']):
        e.decompose()
    return soup.get_text(separator=' ', strip=True)


def extract_boilerpipe(html, **_):
    import boilerpipe.extract as boilerpipe
    text = boilerpipe.Extractor(extractor='ArticleExtractor', html=html)
    text = text.getText()
    return str(text)


def extract_xpath_text(html, **_):
    import lxml.html
    root = lxml.html.fromstring(html)
    text = ' '.join(root.xpath('//body[1]//*[not(name()="script") and not(name()="style")]/text()'))
    text = re.sub(r'(\s+\n\s*)', '\n', text)
    return re.sub(r'[ \t]{2,}', ' ', text)


def extract_news_please(html, **_):
    import newsplease
    return newsplease.NewsPlease.from_html(html, url=None).maintext


def extract_readability(html, **_):
    import readability, html_text
    doc = readability.Document(html)
    text = html_text.extract_text(doc.summary(html_partial=True))
    return text


def extract_go_domdistiller(html, **_):
    from extraction_benchmark.extractors import go_domdistiller
    return go_domdistiller.extract(html)


def extract_inscriptis(html, **_):
    import inscriptis
    text = inscriptis.get_text(html)
    return text


def extract_html_text(html, **_):
    import html_text
    return html_text.extract_text(html)


def extract_resiliparse(html, **_):
    from resiliparse.extract import html2text
    from resiliparse.parse.html import HTMLTree
    return html2text.extract_plain_text(HTMLTree.parse(html),
                                        preserve_formatting=True,
                                        main_content=True,
                                        list_bullets=False,
                                        comments=False,
                                        links=False,
                                        alt_texts=False)


def extract_bte(html, **_):
    from extraction_benchmark.extractors import bte
    return bte.html2text(html)


def extract_trafilatura(html, **_):
    import trafilatura
    return trafilatura.extract(html, include_comments=False)


def extract_justext(html, **_):
    import justext
    article = ' '.join(
        [p.text for p in justext.justext(html, justext.get_stoplist("English"), 50, 200, 0.1, 0.2, 0.2, 200, True)
         if not p.is_boilerplate])
    return article


def extract_goose3(html, **_):
    from goose3 import Goose, configuration
    c = configuration.Configuration()
    c.http_timeout = 5

    with Goose(c) as g:
        article = g.extract(raw_html=html)
        return article.cleaned_text


def extract_lxml_cleaner(html, **_):
    from bs4 import BeautifulSoup
    from lxml.html.clean import Cleaner

    tag_blacklist = [
        # important
        'aside', 'embed', 'footer', 'form', 'head', 'iframe', 'menu', 'object', 'script',
        # other content
        'applet', 'audio', 'canvas', 'figure', 'map', 'picture', 'svg', 'video',
        # secondary
        'area', 'blink', 'button', 'datalist', 'dialog',
        'frame', 'frameset', 'fieldset', 'link', 'input', 'ins', 'label', 'legend',
        'marquee', 'math', 'menuitem', 'nav', 'noscript', 'optgroup', 'option',
        'output', 'param', 'progress', 'rp', 'rt', 'rtc', 'select', 'source',
        'style', 'track', 'template', 'textarea', 'time', 'use',
    ]

    cleaner = Cleaner(
        annoying_tags=False,  # True
        comments=True,
        embedded=False,  # True
        forms=True,  # True
        frames=True,  # True
        javascript=True,
        links=False,
        meta=False,
        page_structure=False,
        processing_instructions=True,
        remove_unknown_tags=False,
        safe_attrs_only=False,
        scripts=True,
        style=False,
        kill_tags=tag_blacklist
    )
    return BeautifulSoup(cleaner.clean_html(html), 'html.parser').get_text(separator=' ', strip=True)


def extract_boilernet(html, **_):
    from extraction_benchmark.extractors import boilernet
    return boilernet.extract(html)


def extract_web2text(html, **_):
    from extraction_benchmark.extractors import web2text
    return web2text.extract(html)


def extract_newspaper3k(html, **_):
    import newspaper
    article = newspaper.Article('')
    article.set_html(html)
    article.parse()
    return article.text


def _get_ensemble_model_list(best_only=False, weighted=False):
    def _ls():
        if best_only or weighted:
            return [
                (extract_goose3, 2 if weighted else 1),
                (extract_readability, 2 if weighted else 1),
                (extract_trafilatura, 1),
                (extract_go_domdistiller, 1),
                (extract_resiliparse, 1),
                (extract_web2text, 1),
                (extract_bte, 1),
                (extract_justext, 1),
                (extract_boilerpipe, 1),
            ]

        return [(m, 1) for m in list_extractors(names_only=False)]

    return zip(*[(m.__name__.replace('extract_', ''), w) for m, w in _ls()])


def extract_ensemble_majority(html, page_id):
    from extraction_benchmark.extractors import ensemble
    models, weights = _get_ensemble_model_list()
    return ensemble.extract_majority_vote(html, page_id, models, weights, int(len(models) * .75))


def extract_ensemble_best(html, page_id):
    from extraction_benchmark.extractors import ensemble
    models, weights = _get_ensemble_model_list(best_only=True)
    return ensemble.extract_majority_vote(html, page_id, models, weights, int(len(models) * .75))


def extract_ensemble_weighted(html, page_id):
    from extraction_benchmark.extractors import ensemble
    models, weights = _get_ensemble_model_list(best_only=True, weighted=True)
    return ensemble.extract_majority_vote(html, page_id, models, weights, int(len(models) * .75))


def list_extractors(names_only=True, include_ensembles=False):
    """
    Get a list of all supported extraction systems.

    :param names_only: only return a list of strings (otherwise return extractor routines)
    :param include_ensembles: include ensemble extractors in the list
    :return: list of extractor names or functions
    """
    return [(n.replace('extract_', '') if names_only else m) for n, m in globals().items()
            if n.startswith('extract_') and (not n.startswith('extract_ensemble') or include_ensembles)]
