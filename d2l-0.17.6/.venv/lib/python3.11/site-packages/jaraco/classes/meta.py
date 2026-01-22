"""
meta.py

Some useful metaclasses.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any


class LeafClassesMeta(type):
    """
    A metaclass for classes that keeps track of all of them that
    aren't base classes.

    >>> Parent = LeafClassesMeta('MyParentClass', (), {})
    >>> Parent in Parent._leaf_classes
    True
    >>> Child = LeafClassesMeta('MyChildClass', (Parent,), {})
    >>> Child in Parent._leaf_classes
    True
    >>> Parent in Parent._leaf_classes
    False

    >>> Other = LeafClassesMeta('OtherClass', (), {})
    >>> Parent in Other._leaf_classes
    False
    >>> len(Other._leaf_classes)
    1
    """

    _leaf_classes: set[type[Any]]

    def __init__(
        cls,
        name: str,
        bases: tuple[type[object], ...],
        attrs: dict[str, object],
    ) -> None:
        if not hasattr(cls, '_leaf_classes'):
            cls._leaf_classes = set()
        leaf_classes = getattr(cls, '_leaf_classes')
        leaf_classes.add(cls)
        # remove any base classes
        leaf_classes -= set(bases)


class TagRegistered(type):
    """
    As classes of this metaclass are created, they keep a registry in the
    base class of all classes by a class attribute, indicated by attr_name.

    >>> FooObject = TagRegistered('FooObject', (), dict(tag='foo'))
    >>> FooObject._registry['foo'] is FooObject
    True
    >>> BarObject = TagRegistered('Barobject', (FooObject,), dict(tag='bar'))
    >>> FooObject._registry is BarObject._registry
    True
    >>> len(FooObject._registry)
    2

    '...' below should be 'jaraco.classes' but for pytest-dev/pytest#3396
    >>> FooObject._registry['bar']
    <class '....meta.Barobject'>
    """

    attr_name = 'tag'

    def __init__(
        cls,
        name: str,
        bases: tuple[type[object], ...],
        namespace: dict[str, object],
    ) -> None:
        super(TagRegistered, cls).__init__(name, bases, namespace)
        if not hasattr(cls, '_registry'):
            cls._registry = {}
        meta = cls.__class__
        attr = getattr(cls, meta.attr_name, None)
        if attr:
            cls._registry[attr] = cls
