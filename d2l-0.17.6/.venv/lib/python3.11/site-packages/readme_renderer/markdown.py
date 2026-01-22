# Copyright 2014 Donald Stufft
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import warnings
from typing import cast, Any, Dict, Callable, Match, Optional

from html import unescape

import pygments
import pygments.lexers
import pygments.formatters

from .clean import clean

_EXTRA_WARNING = (
    "Markdown renderers are not available. "
    "Install 'readme_renderer[md]' to enable Markdown rendering."
)

try:
    import cmarkgfm
    from cmarkgfm.cmark import Options as cmarkgfmOptions
    variants: Dict[str, Callable[[str], str]] = {
        "GFM": lambda raw: cast(str, cmarkgfm.github_flavored_markdown_to_html(
            raw, options=cmarkgfmOptions.CMARK_OPT_UNSAFE
        )),
        "CommonMark": lambda raw: cast(str, cmarkgfm.markdown_to_html(
            raw, options=cmarkgfmOptions.CMARK_OPT_UNSAFE
        )),
    }
except ImportError:
    warnings.warn(_EXTRA_WARNING)
    variants = {}

# Make code fences with `python` as the language default to highlighting as
# Python 3.
_LANG_ALIASES = {
    'python': 'python3',
}


def render(
    raw: str,
    variant: str = "GFM",
    **kwargs: Any
) -> Optional[str]:
    if not variants:
        warnings.warn(_EXTRA_WARNING)
        return None

    renderer = variants.get(variant)

    if not renderer:
        return None

    rendered = renderer(raw)

    if not rendered:
        return None

    highlighted = _highlight(rendered)
    cleaned = clean(highlighted)
    return cleaned


def _highlight(html: str) -> str:
    """Syntax-highlights HTML-rendered Markdown.

    Plucks sections to highlight that conform the the GitHub fenced code info
    string as defined at https://github.github.com/gfm/#info-string.

    Args:
        html (str): The rendered HTML.

    Returns:
        str: The HTML with Pygments syntax highlighting applied to all code
            blocks.
    """

    formatter = pygments.formatters.HtmlFormatter(nowrap=True)

    code_expr = re.compile(
        # cmarkgfm<0.6.0: <pre><code class="language-python">print('hello')</code></pre>
        # cmarkgfm>=0.6.0: <pre lang="python"><code>print('hello')</code></pre>
        r'(<pre>(?P<in_code><code) class="language-|<pre lang=")(?P<lang>[^"]+?)">'
        '(?(in_code)|<code>)(?P<code>.+?)'
        r'</code></pre>', re.DOTALL)

    def replacer(match: Match[Any]) -> str:
        try:
            lang = match.group('lang')
            lang = _LANG_ALIASES.get(lang, lang)
            lexer = pygments.lexers.get_lexer_by_name(lang)
        except ValueError:
            lexer = pygments.lexers.TextLexer()

        code = match.group('code')

        # Decode html entities in the code. cmark tries to be helpful and
        # translate '"' to '&quot;', but it confuses pygments. Pygments will
        # escape any html entities when re-writing the code, and we run
        # everything through bleach after.
        code = unescape(code)

        highlighted = pygments.highlight(code, lexer, formatter)

        return f'<pre lang="{lang}">{highlighted}</pre>'

    result = code_expr.sub(replacer, html)

    return result
