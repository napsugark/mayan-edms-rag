from django.core.exceptions import ValidationError as DjangoValidationError

from rest_framework.exceptions import ValidationError
from rest_framework.views import exception_handler


def mayan_exception_handler(exc, context):
    """
    Custom REST API exception handler to ensure all Django ValidationError
    exceptions are converted into HTTP 400 errors and not HTTP 500 errors.
    """
    if isinstance(exc, DjangoValidationError):
        try:
            exc = ValidationError(code=exc.code, detail=exc.message)
        except AttributeError:
            exc = ValidationError(detail=exc.messages)

    return exception_handler(context=context, exc=exc)
