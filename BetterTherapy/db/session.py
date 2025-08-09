from functools import wraps
from .connection import SessionLocal


def session(func):
    @wraps(func)
    def session_wrapper(*args, **kwargs):
        with SessionLocal() as session:
            return func(session, *args, **kwargs)

    return session_wrapper
