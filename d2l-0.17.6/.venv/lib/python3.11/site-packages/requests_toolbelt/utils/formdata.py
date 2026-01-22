# -*- coding: utf-8 -*-
"""Implementation of nested form-data encoding function(s)."""
from .._compat import basestring
from .._compat import urlencode as _urlencode


__all__ = ('urlencode',)


def urlencode(query, *args, **kwargs):
    """Handle nested form-data queries and serialize them appropriately.

    There are times when a website expects a nested form data query to be sent
    but, the standard library's urlencode function does not appropriately
    handle the nested structures. In that case, you need this function which
    will flatten the structure first and then properly encode it for you.

    When using this to send data in the body of a request, make sure you
    specify the appropriate Content-Type header for the request.

    .. code-block:: python

        import requests
        from requests_toolbelt.utils import formdata

        query = {
           'my_dict': {
               'foo': 'bar',
               'biz': 'baz",
            },
            'a': 'b',
        }

        resp = requests.get(url, params=formdata.urlencode(query))
        # or
        resp = requests.post(
            url,
            data=formdata.urlencode(query),
            headers={
                'Content-Type': 'application/x-www-form-urlencoded'
            },
        )

    Similarly, you can specify a list of nested tuples, e.g.,

    .. code-block:: python

        import requests
        from requests_toolbelt.utils import formdata

        query = [
            ('my_list', [
                ('foo', 'bar'),
                ('biz', 'baz'),
            ]),
            ('a', 'b'),
        ]

        resp = requests.get(url, params=formdata.urlencode(query))
        # or
        resp = requests.post(
            url,
            data=formdata.urlencode(query),
            headers={
                'Content-Type': 'application/x-www-form-urlencoded'
            },
        )

    For additional parameter and return information, see the official
    `urlencode`_ documentation.

    .. _urlencode:
        https://docs.python.org/3/library/urllib.parse.html#urllib.parse.urlencode
    """
    expand_classes = (dict, list, tuple)
    original_query_list = _to_kv_list(query)

    if not all(_is_two_tuple(i) for i in original_query_list):
        raise ValueError("Expected query to be able to be converted to a "
                         "list comprised of length 2 tuples.")

    query_list = original_query_list
    while any(isinstance(v, expand_classes) for _, v in query_list):
        query_list = _expand_query_values(query_list)

    return _urlencode(query_list, *args, **kwargs)


def _to_kv_list(dict_or_list):
    if hasattr(dict_or_list, 'items'):
        return list(dict_or_list.items())
    return dict_or_list


def _is_two_tuple(item):
    return isinstance(item, (list, tuple)) and len(item) == 2


def _expand_query_values(original_query_list):
    query_list = []
    for key, value in original_query_list:
        if isinstance(value, basestring):
            query_list.append((key, value))
        else:
            key_fmt = key + '[%s]'
            value_list = _to_kv_list(value)
            query_list.extend((key_fmt % k, v) for k, v in value_list)
    return query_list
