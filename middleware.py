from fastapi import Request
from contextvars import ContextVar


REQUEST_CTX_KEY = "request_context"
_request_ctx_var: ContextVar[str | None] = ContextVar(REQUEST_CTX_KEY, default=None)


async def request_context_middleware(request: Request, call_next):
    """ Middleware to set request context for use in other parts of the app.
    example: request_context = _request_ctx_var.get()
    """
    try:
        request_ctx = _request_ctx_var.set(request)
        response = await call_next(request)
        _request_ctx_var.reset(request_ctx)
        return response
    except Exception as e:
        raise e
