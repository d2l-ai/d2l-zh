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

from typing import Dict, Optional, Set

import nh3


ALLOWED_TAGS = {
    # Bleach Defaults
    "a", "abbr", "acronym", "b", "blockquote", "code", "em", "i", "li", "ol",
    "strong", "ul",

    # Custom Additions
    "br", "caption", "cite", "col", "colgroup", "dd", "del", "details", "div",
    "dl", "dt", "h1", "h2", "h3", "h4", "h5", "h6", "hr", "img", "p", "pre",
    "span", "sub", "summary", "sup", "table", "tbody", "td", "th", "thead",
    "tr", "tt", "kbd", "var", "input", "section", "aside", "nav", "figure",
    "figcaption", "picture",
}

ALLOWED_ATTRIBUTES = {
    # Bleach Defaults
    "a": {"href", "title"},
    "abbr": {"title"},
    "acronym": {"title"},

    # Custom Additions
    "*": {"id"},
    "hr": {"class"},
    "img": {"src", "width", "height", "alt", "align", "class"},
    "span": {"class"},
    "th": {"align", "class"},
    "td": {"align", "colspan", "rowspan"},
    "div": {"align", "class"},
    "h1": {"align"},
    "h2": {"align"},
    "h3": {"align"},
    "h4": {"align"},
    "h5": {"align"},
    "h6": {"align"},
    "code": {"class"},
    "p": {"align", "class"},
    "pre": {"lang"},
    "ol": {"start"},
    "input": {"type", "checked", "disabled"},
    "aside": {"class"},
    "dd": {"class"},
    "dl": {"class"},
    "dt": {"class"},
    "ul": {"class"},
    "nav": {"class"},
    "figure": {"class"},
}


def clean(
    html: str,
    tags: Optional[Set[str]] = None,
    attributes: Optional[Dict[str, Set[str]]] = None
) -> Optional[str]:
    if tags is None:
        tags = ALLOWED_TAGS
    if attributes is None:
        attributes = ALLOWED_ATTRIBUTES

    try:
        cleaned = nh3.clean(
            html,
            tags=ALLOWED_TAGS,
            attributes=ALLOWED_ATTRIBUTES,
            link_rel="nofollow",
            url_schemes={"http", "https", "mailto"},
        )

        return cleaned
    except ValueError:
        return None
