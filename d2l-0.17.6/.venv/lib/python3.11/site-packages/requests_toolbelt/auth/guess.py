# -*- coding: utf-8 -*-
"""The module containing the code for GuessAuth."""
from requests import auth
from requests import cookies

from . import _digest_auth_compat as auth_compat, http_proxy_digest


class GuessAuth(auth.AuthBase):
    """Guesses the auth type by the WWW-Authentication header."""
    def __init__(self, username, password):
        self.username = username
        self.password = password
        self.auth = None
        self.pos = None

    def _handle_basic_auth_401(self, r, kwargs):
        if self.pos is not None:
            r.request.body.seek(self.pos)

        # Consume content and release the original connection
        # to allow our new request to reuse the same one.
        r.content
        r.raw.release_conn()
        prep = r.request.copy()
        if not hasattr(prep, '_cookies'):
            prep._cookies = cookies.RequestsCookieJar()
        cookies.extract_cookies_to_jar(prep._cookies, r.request, r.raw)
        prep.prepare_cookies(prep._cookies)

        self.auth = auth.HTTPBasicAuth(self.username, self.password)
        prep = self.auth(prep)
        _r = r.connection.send(prep, **kwargs)
        _r.history.append(r)
        _r.request = prep

        return _r

    def _handle_digest_auth_401(self, r, kwargs):
        self.auth = auth_compat.HTTPDigestAuth(self.username, self.password)
        try:
            self.auth.init_per_thread_state()
        except AttributeError:
            # If we're not on requests 2.8.0+ this method does not exist and
            # is not relevant.
            pass

        # Check that the attr exists because much older versions of requests
        # set this attribute lazily. For example:
        # https://github.com/kennethreitz/requests/blob/33735480f77891754304e7f13e3cdf83aaaa76aa/requests/auth.py#L59
        if (hasattr(self.auth, 'num_401_calls') and
                self.auth.num_401_calls is None):
            self.auth.num_401_calls = 1
        # Digest auth would resend the request by itself. We can take a
        # shortcut here.
        return self.auth.handle_401(r, **kwargs)

    def handle_401(self, r, **kwargs):
        """Resends a request with auth headers, if needed."""

        www_authenticate = r.headers.get('www-authenticate', '').lower()

        if 'basic' in www_authenticate:
            return self._handle_basic_auth_401(r, kwargs)

        if 'digest' in www_authenticate:
            return self._handle_digest_auth_401(r, kwargs)

    def __call__(self, request):
        if self.auth is not None:
            return self.auth(request)

        try:
            self.pos = request.body.tell()
        except AttributeError:
            pass

        request.register_hook('response', self.handle_401)
        return request


class GuessProxyAuth(GuessAuth):
    """
    Guesses the auth type by WWW-Authentication and Proxy-Authentication
    headers
    """
    def __init__(self, username=None, password=None,
                 proxy_username=None, proxy_password=None):
        super(GuessProxyAuth, self).__init__(username, password)
        self.proxy_username = proxy_username
        self.proxy_password = proxy_password
        self.proxy_auth = None

    def _handle_basic_auth_407(self, r, kwargs):
        if self.pos is not None:
            r.request.body.seek(self.pos)

        r.content
        r.raw.release_conn()
        prep = r.request.copy()
        if not hasattr(prep, '_cookies'):
            prep._cookies = cookies.RequestsCookieJar()
        cookies.extract_cookies_to_jar(prep._cookies, r.request, r.raw)
        prep.prepare_cookies(prep._cookies)

        self.proxy_auth = auth.HTTPProxyAuth(self.proxy_username,
                                             self.proxy_password)
        prep = self.proxy_auth(prep)
        _r = r.connection.send(prep, **kwargs)
        _r.history.append(r)
        _r.request = prep

        return _r

    def _handle_digest_auth_407(self, r, kwargs):
        self.proxy_auth = http_proxy_digest.HTTPProxyDigestAuth(
            username=self.proxy_username,
            password=self.proxy_password)

        try:
            self.auth.init_per_thread_state()
        except AttributeError:
            pass

        return self.proxy_auth.handle_407(r, **kwargs)

    def handle_407(self, r, **kwargs):
        proxy_authenticate = r.headers.get('Proxy-Authenticate', '').lower()

        if 'basic' in proxy_authenticate:
            return self._handle_basic_auth_407(r, kwargs)

        if 'digest' in proxy_authenticate:
            return self._handle_digest_auth_407(r, kwargs)

    def __call__(self, request):
        if self.proxy_auth is not None:
            request = self.proxy_auth(request)

        try:
            self.pos = request.body.tell()
        except AttributeError:
            pass

        request.register_hook('response', self.handle_407)
        return super(GuessProxyAuth, self).__call__(request)
