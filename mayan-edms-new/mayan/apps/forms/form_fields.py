from django.forms import ModelChoiceField  # NOQA
from django.forms.fields import *  # NOQA
from django.forms.fields import ChoiceField, MultipleChoiceField
from django.forms.models import ModelMultipleChoiceField

from .field_mixins import (
    FormFieldMixinFilteredQueryset, ModelFieldMixinFilteredQuerySet
)


class FormFieldFilteredModelChoice(
    FormFieldMixinFilteredQueryset, ChoiceField
):
    """Single selection filtered model choice field."""


class FormFieldFilteredModelChoiceMultiple(
    FormFieldMixinFilteredQueryset, MultipleChoiceField
):
    """Multiple selection filtered model choice field."""


class ModelFormFieldFilteredModelMultipleChoice(
    ModelFieldMixinFilteredQuerySet, ModelMultipleChoiceField
):
    """Multiple selection filtered model choice field."""
