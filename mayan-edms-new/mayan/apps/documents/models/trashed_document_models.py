from ..managers import TrashCanManager

from .document_models import Document
from .trashed_document_model_mixins import TrashedDocumentBusinessLogicMixin

__all__ = ('TrashedDocument',)


class TrashedDocument(TrashedDocumentBusinessLogicMixin, Document):
    objects = TrashCanManager()

    class Meta:
        proxy = True
