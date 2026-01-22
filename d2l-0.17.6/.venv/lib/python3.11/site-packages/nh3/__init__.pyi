from typing import Callable, Dict, Optional, Set

ALLOWED_TAGS: Set[str]
ALLOWED_ATTRIBUTES: Dict[str, Set[str]]
ALLOWED_URL_SCHEMES: Set[str]

class Cleaner:
    def __init__(
        self,
        tags: Optional[Set[str]] = None,
        clean_content_tags: Optional[Set[str]] = None,
        attributes: Optional[Dict[str, Set[str]]] = None,
        attribute_filter: Optional[Callable[[str, str, str], Optional[str]]] = None,
        strip_comments: bool = True,
        link_rel: Optional[str] = "noopener noreferrer",
        generic_attribute_prefixes: Optional[Set[str]] = None,
        tag_attribute_values: Optional[Dict[str, Dict[str, Set[str]]]] = None,
        set_tag_attribute_values: Optional[Dict[str, Dict[str, str]]] = None,
        url_schemes: Optional[Set[str]] = None,
        allowed_classes: Optional[Dict[str, Set[str]]] = None,
        filter_style_properties: Optional[Set[str]] = None,
    ) -> None: ...
    def clean(self, html: str) -> str: ...

def clean(
    html: str,
    tags: Optional[Set[str]] = None,
    clean_content_tags: Optional[Set[str]] = None,
    attributes: Optional[Dict[str, Set[str]]] = None,
    attribute_filter: Optional[Callable[[str, str, str], Optional[str]]] = None,
    strip_comments: bool = True,
    link_rel: Optional[str] = "noopener noreferrer",
    generic_attribute_prefixes: Optional[Set[str]] = None,
    tag_attribute_values: Optional[Dict[str, Dict[str, Set[str]]]] = None,
    set_tag_attribute_values: Optional[Dict[str, Dict[str, str]]] = None,
    url_schemes: Optional[Set[str]] = None,
    allowed_classes: Optional[Dict[str, Set[str]]] = None,
    filter_style_properties: Optional[Set[str]] = None,
) -> str: ...
def clean_text(html: str) -> str: ...
def is_html(html: str) -> bool: ...
