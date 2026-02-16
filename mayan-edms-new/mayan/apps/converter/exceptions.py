class ConvertError(Exception):
    """
    Base exception for all converter app exceptions.
    """


class AppImageError(ConvertError):
    """
    Exception to allow app specific error codes. These are errors that might
    need additional handling and that are not just a generic unknown format
    error.
    """
    def __init__(self, error_name, details=None):
        self.details = details
        self.error_name = error_name
        super().__init__()

    def __str__(self):
        return (
            'Error name: {}'.format(
                repr(self.error_name)
            )
        )


class InvalidOfficeFormat(ConvertError):
    """
    Raised by the file type introspection code to signal that the file is not
    a office format file and that LibreOffice will not be used to process
    it.
    """


class LayerError(ConvertError):
    """
    Raise by the layer class when attempting to create a transformation in
    a layer for which it was not registered.
    """


class OfficeConversionError(ConvertError):
    """
    Used to encapsulate errors while executing LibreOffice or when
    LibreOffice is now available.
    """


class PageCountError(ConvertError):
    """
    Raised when an error is encountered while determining the page count of a
    file.
    """


class UnknownFileFormat(ConvertError):
    """
    Raised when the converter backend can't understand a file.
    """


class UnkownConvertError(ConvertError):
    """
    Raised when an error is found but there is no discernible way to
    identify the kind of error.
    """
