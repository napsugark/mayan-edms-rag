class TemplatingError(Exception):
    """
    Base exception for all templating app exceptions.
    """


class DangerousTagError(TemplatingError):
    """
    Raised when a template tries to execute a dangerous tag.
    """
