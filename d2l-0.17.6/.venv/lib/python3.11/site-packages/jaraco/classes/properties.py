from __future__ import annotations

from typing import TYPE_CHECKING, Generic, TypeVar, cast, overload

_T = TypeVar('_T')
_U = TypeVar('_U')

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any, Protocol

    from typing_extensions import Self, TypeAlias

    # TODO(coherent-oss/granary#4): Migrate to PEP 695 by 2027-10.
    _GetterCallable: TypeAlias = Callable[..., _T]
    _GetterClassMethod: TypeAlias = classmethod[Any, [], _T]

    _SetterCallable: TypeAlias = Callable[[type[Any], _T], None]
    _SetterClassMethod: TypeAlias = classmethod[Any, [_T], None]

    class _ClassPropertyAttribute(Protocol[_T]):
        def __get__(self, obj: object, objtype: type[Any] | None = None) -> _T: ...

        def __set__(self, obj: object, value: _T) -> None: ...


class NonDataProperty(Generic[_T, _U]):
    """Much like the property builtin, but only implements __get__,
    making it a non-data property, and can be subsequently reset.

    See http://users.rcn.com/python/download/Descriptor.htm for more
    information.

    >>> class X(object):
    ...   @NonDataProperty
    ...   def foo(self):
    ...     return 3
    >>> x = X()
    >>> x.foo
    3
    >>> x.foo = 4
    >>> x.foo
    4

    '...' below should be 'jaraco.classes' but for pytest-dev/pytest#3396
    >>> X.foo
    <....properties.NonDataProperty object at ...>
    """

    def __init__(self, fget: Callable[[_T], _U]) -> None:
        assert fget is not None, "fget cannot be none"
        assert callable(fget), "fget must be callable"
        self.fget = fget

    @overload
    def __get__(
        self,
        obj: None,
        objtype: None,
    ) -> Self: ...

    @overload
    def __get__(
        self,
        obj: _T,
        objtype: type[_T] | None = None,
    ) -> _U: ...

    def __get__(
        self,
        obj: _T | None,
        objtype: type[_T] | None = None,
    ) -> Self | _U:
        if obj is None:
            return self
        return self.fget(obj)


class classproperty(Generic[_T]):
    """
    Like @property but applies at the class level.


    >>> class X(metaclass=classproperty.Meta):
    ...   val = None
    ...   @classproperty
    ...   def foo(cls):
    ...     return cls.val
    ...   @foo.setter
    ...   def foo(cls, val):
    ...     cls.val = val
    >>> X.foo
    >>> X.foo = 3
    >>> X.foo
    3
    >>> x = X()
    >>> x.foo
    3
    >>> X.foo = 4
    >>> x.foo
    4

    Setting the property on an instance affects the class.

    >>> x.foo = 5
    >>> x.foo
    5
    >>> X.foo
    5
    >>> vars(x)
    {}
    >>> X().foo
    5

    Attempting to set an attribute where no setter was defined
    results in an AttributeError:

    >>> class GetOnly(metaclass=classproperty.Meta):
    ...   @classproperty
    ...   def foo(cls):
    ...     return 'bar'
    >>> GetOnly.foo = 3
    Traceback (most recent call last):
    ...
    AttributeError: can't set attribute

    It is also possible to wrap a classmethod or staticmethod in
    a classproperty.

    >>> class Static(metaclass=classproperty.Meta):
    ...   @classproperty
    ...   @classmethod
    ...   def foo(cls):
    ...     return 'foo'
    ...   @classproperty
    ...   @staticmethod
    ...   def bar():
    ...     return 'bar'
    >>> Static.foo
    'foo'
    >>> Static.bar
    'bar'

    *Legacy*

    For compatibility, if the metaclass isn't specified, the
    legacy behavior will be invoked.

    >>> class X:
    ...   val = None
    ...   @classproperty
    ...   def foo(cls):
    ...     return cls.val
    ...   @foo.setter
    ...   def foo(cls, val):
    ...     cls.val = val
    >>> X.foo
    >>> X.foo = 3
    >>> X.foo
    3
    >>> x = X()
    >>> x.foo
    3
    >>> X.foo = 4
    >>> x.foo
    4

    Note, because the metaclass was not specified, setting
    a value on an instance does not have the intended effect.

    >>> x.foo = 5
    >>> x.foo
    5
    >>> X.foo  # should be 5
    4
    >>> vars(x)  # should be empty
    {'foo': 5}
    >>> X().foo  # should be 5
    4
    """

    fget: _ClassPropertyAttribute[_GetterClassMethod[_T]]
    fset: _ClassPropertyAttribute[_SetterClassMethod[_T] | None]

    class Meta(type):
        def __setattr__(self, key: str, value: object) -> None:
            obj = self.__dict__.get(key, None)
            if type(obj) is classproperty:
                return obj.__set__(self, value)
            return super().__setattr__(key, value)

    def __init__(
        self,
        fget: _GetterCallable[_T] | _GetterClassMethod[_T],
        fset: _SetterCallable[_T] | _SetterClassMethod[_T] | None = None,
    ) -> None:
        self.fget = self._ensure_method(fget)
        self.fset = fset  # type: ignore[assignment] # Corrected in the next line.
        fset and self.setter(fset)

    def __get__(self, instance: object, owner: type[object] | None = None) -> _T:
        return self.fget.__get__(None, owner)()

    def __set__(self, owner: object, value: _T) -> None:
        if not self.fset:
            raise AttributeError("can't set attribute")
        if type(owner) is not classproperty.Meta:
            owner = type(owner)
        return self.fset.__get__(None, cast('type[object]', owner))(value)

    def setter(self, fset: _SetterCallable[_T] | _SetterClassMethod[_T]) -> Self:
        self.fset = self._ensure_method(fset)
        return self

    @overload
    @classmethod
    def _ensure_method(
        cls,
        fn: _GetterCallable[_T] | _GetterClassMethod[_T],
    ) -> _GetterClassMethod[_T]: ...

    @overload
    @classmethod
    def _ensure_method(
        cls,
        fn: _SetterCallable[_T] | _SetterClassMethod[_T],
    ) -> _SetterClassMethod[_T]: ...

    @classmethod
    def _ensure_method(
        cls,
        fn: _GetterCallable[_T]
        | _GetterClassMethod[_T]
        | _SetterCallable[_T]
        | _SetterClassMethod[_T],
    ) -> _GetterClassMethod[_T] | _SetterClassMethod[_T]:
        """
        Ensure fn is a classmethod or staticmethod.
        """
        needs_method = not isinstance(fn, (classmethod, staticmethod))
        return classmethod(fn) if needs_method else fn  # type: ignore[arg-type,return-value]
