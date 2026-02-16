class SettingsException(Exception):
    """
    Base exception for the smart_settings app.
    """


class SettingsExceptionRevert(SettingsException):
    """
    Raised when attempting to revert an setting's value.
    """
