class SourceException(Exception):
    """Base sources warning"""


class SourceActionException(SourceException):
    """Raised when a source does not have the request action."""


class SourceActionExceptionUnknown(SourceActionException):
    """Raised when a source does not have the request action."""


class SourceActionExceptionInterface(SourceActionException):
    """Base exception for all things related to the action interfaces."""


class SourceActionExceptionInterfaceArgumentMissing(SourceActionExceptionInterface):
    """
    Raised when an action does not supply a required interface argument
    which does not have a default value.
    """


class SourceActionExceptionInterfaceUnknown(SourceActionExceptionInterface):
    """Raised when an action does not support the specified interface."""
