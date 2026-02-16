from django.utils.translation import gettext


class DocumentCheckoutError(Exception):
    """
    Base checkout exception.
    """


class DocumentNotCheckedOut(DocumentCheckoutError):
    """
    Raised when trying to checkin a document that is not checked out.
    """
    def __str__(self):
        return gettext('Document not checked out.')


class DocumentAlreadyCheckedOut(DocumentCheckoutError):
    """
    Raised when trying to checkout an already checkedout document.
    """
    def __str__(self):
        return gettext(message='Document already checked out.')


class NewDocumentFileNotAllowed(DocumentCheckoutError):
    """
    Uploading new versions for this document is not allowed.
    """
