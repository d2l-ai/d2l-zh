"""This module provides functions for dumping information about responses."""
import collections

from requests import compat


__all__ = ('dump_response', 'dump_all')

HTTP_VERSIONS = {
    9: b'0.9',
    10: b'1.0',
    11: b'1.1',
}

_PrefixSettings = collections.namedtuple('PrefixSettings',
                                         ['request', 'response'])


class PrefixSettings(_PrefixSettings):
    def __new__(cls, request, response):
        request = _coerce_to_bytes(request)
        response = _coerce_to_bytes(response)
        return super(PrefixSettings, cls).__new__(cls, request, response)


def _get_proxy_information(response):
    if getattr(response.connection, 'proxy_manager', False):
        proxy_info = {}
        request_url = response.request.url
        if request_url.startswith('https://'):
            proxy_info['method'] = 'CONNECT'

        proxy_info['request_path'] = request_url
        return proxy_info
    return None


def _format_header(name, value):
    return (_coerce_to_bytes(name) + b': ' + _coerce_to_bytes(value) +
            b'\r\n')


def _build_request_path(url, proxy_info):
    uri = compat.urlparse(url)
    proxy_url = proxy_info.get('request_path')
    if proxy_url is not None:
        request_path = _coerce_to_bytes(proxy_url)
        return request_path, uri

    request_path = _coerce_to_bytes(uri.path)
    if uri.query:
        request_path += b'?' + _coerce_to_bytes(uri.query)

    return request_path, uri


def _dump_request_data(request, prefixes, bytearr, proxy_info=None):
    if proxy_info is None:
        proxy_info = {}

    prefix = prefixes.request
    method = _coerce_to_bytes(proxy_info.pop('method', request.method))
    request_path, uri = _build_request_path(request.url, proxy_info)

    # <prefix><METHOD> <request-path> HTTP/1.1
    bytearr.extend(prefix + method + b' ' + request_path + b' HTTP/1.1\r\n')

    # <prefix>Host: <request-host> OR host header specified by user
    headers = request.headers.copy()
    host_header = _coerce_to_bytes(headers.pop('Host', uri.netloc))
    bytearr.extend(prefix + b'Host: ' + host_header + b'\r\n')

    for name, value in headers.items():
        bytearr.extend(prefix + _format_header(name, value))

    bytearr.extend(prefix + b'\r\n')
    if request.body:
        if isinstance(request.body, compat.basestring):
            bytearr.extend(prefix + _coerce_to_bytes(request.body))
        else:
            # In the event that the body is a file-like object, let's not try
            # to read everything into memory.
            bytearr.extend(b'<< Request body is not a string-like type >>')
        bytearr.extend(b'\r\n')
    bytearr.extend(b'\r\n')


def _dump_response_data(response, prefixes, bytearr):
    prefix = prefixes.response
    # Let's interact almost entirely with urllib3's response
    raw = response.raw

    # Let's convert the version int from httplib to bytes
    version_str = HTTP_VERSIONS.get(raw.version, b'?')

    # <prefix>HTTP/<version_str> <status_code> <reason>
    bytearr.extend(prefix + b'HTTP/' + version_str + b' ' +
                   str(raw.status).encode('ascii') + b' ' +
                   _coerce_to_bytes(response.reason) + b'\r\n')

    headers = raw.headers
    for name in headers.keys():
        for value in headers.getlist(name):
            bytearr.extend(prefix + _format_header(name, value))

    bytearr.extend(prefix + b'\r\n')

    bytearr.extend(response.content)


def _coerce_to_bytes(data):
    if not isinstance(data, bytes) and hasattr(data, 'encode'):
        data = data.encode('utf-8')
    # Don't bail out with an exception if data is None
    return data if data is not None else b''


def dump_response(response, request_prefix=b'< ', response_prefix=b'> ',
                  data_array=None):
    """Dump a single request-response cycle's information.

    This will take a response object and dump only the data that requests can
    see for that single request-response cycle.

    Example::

        import requests
        from requests_toolbelt.utils import dump

        resp = requests.get('https://api.github.com/users/sigmavirus24')
        data = dump.dump_response(resp)
        print(data.decode('utf-8'))

    :param response:
        The response to format
    :type response: :class:`requests.Response`
    :param request_prefix: (*optional*)
        Bytes to prefix each line of the request data
    :type request_prefix: :class:`bytes`
    :param response_prefix: (*optional*)
        Bytes to prefix each line of the response data
    :type response_prefix: :class:`bytes`
    :param data_array: (*optional*)
        Bytearray to which we append the request-response cycle data
    :type data_array: :class:`bytearray`
    :returns: Formatted bytes of request and response information.
    :rtype: :class:`bytearray`
    """
    data = data_array if data_array is not None else bytearray()
    prefixes = PrefixSettings(request_prefix, response_prefix)

    if not hasattr(response, 'request'):
        raise ValueError('Response has no associated request')

    proxy_info = _get_proxy_information(response)
    _dump_request_data(response.request, prefixes, data,
                       proxy_info=proxy_info)
    _dump_response_data(response, prefixes, data)
    return data


def dump_all(response, request_prefix=b'< ', response_prefix=b'> '):
    """Dump all requests and responses including redirects.

    This takes the response returned by requests and will dump all
    request-response pairs in the redirect history in order followed by the
    final request-response.

    Example::

        import requests
        from requests_toolbelt.utils import dump

        resp = requests.get('https://httpbin.org/redirect/5')
        data = dump.dump_all(resp)
        print(data.decode('utf-8'))

    :param response:
        The response to format
    :type response: :class:`requests.Response`
    :param request_prefix: (*optional*)
        Bytes to prefix each line of the request data
    :type request_prefix: :class:`bytes`
    :param response_prefix: (*optional*)
        Bytes to prefix each line of the response data
    :type response_prefix: :class:`bytes`
    :returns: Formatted bytes of request and response information.
    :rtype: :class:`bytearray`
    """
    data = bytearray()

    history = list(response.history[:])
    history.append(response)

    for response in history:
        dump_response(response, request_prefix, response_prefix, data)

    return data
