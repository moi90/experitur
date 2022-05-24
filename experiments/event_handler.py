from functools import wraps
from typing import Any, Callable, TypeVar

_TFunc = TypeVar("_TFunc", bound=Callable[..., Any])


class EventHandler:
    def __init__(self, f) -> None:
        super().__init__()
        self.f = f

    def __call__(self, *args, **kwargs):
        print("wrapper", *args, **kwargs)
        return self.f(*args, **kwargs)

    def handle(self, *args, **kwargs):
        ...


def decorator(f: _TFunc) -> _TFunc:
    return EventHandler(f)


class A:
    @decorator
    def on_success(self, handler: Callable[[int], int]):
        """Some documentation."""
        pass


a = A()


@a.on_success
def a_on_success(i: int):
    return i + 1


a.on_success.handle()
