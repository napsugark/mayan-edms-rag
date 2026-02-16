class DynamicSearchException(Exception):
    """
    Base exception for the app.
    """


class DynamicSearchBackendException(DynamicSearchException):
    """
    Encapsulate search backend specific errors.
    """


class DynamicSearchInterpreterError(DynamicSearchException):
    """
    Raised when there is an error in a search interpreter.
    """


class DynamicSearchInterpreterUnknownSearchType(
    DynamicSearchInterpreterError
):
    """
    Raised when there is not search interpreter that matches a query.
    """


class DynamicSearchModelException(DynamicSearchException):
    """
    Used to raise error related to search models and search fields.
    """


class DynamicSearchQueryError(DynamicSearchException):
    """
    Raised when there is an error in a search query.
    """


class DynamicSearchRetry(DynamicSearchException):
    """
    Exception to encapsulate backend specific errors that should be retried.
    """


class DynamicSearchScopedQueryError(DynamicSearchException):
    """
    Raised when there is an error in a scoped query.
    """


class DynamicSearchValueTransformationError(DynamicSearchException):
    """
    Raised when it is not possible to transform a value.
    """
