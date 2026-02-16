from django.utils.translation import gettext_lazy as _

from mayan.apps.backends.class_mixins import DynamicFormBackendMixin
from mayan.apps.backends.classes import ModelBaseBackend

from ..exceptions import SourceActionExceptionUnknown

from .mixins import SourceBackendMixinSourceMetadata


class SourceBackendBase:
    """
    Base class for the source backends.
    """
    action_class_list = None

    def callback_post_document_create(self, document, **kwargs):
        """
        Callback to execute when a document is created.
        """
        return

    def callback_post_document_file_create(self, document_file, **kwargs):
        """
        Callback to execute when a document file is first created.
        """
        return

    def callback_post_document_file_upload(self, document_file, **kwargs):
        """
        Callback to execute when a document file is fully uploaded.
        """
        return

    def clean(self):
        """
        Optional method to validate backend data before saving.
        """

    def create(self):
        """
        Called after the source model's .save() method for new
        instances.
        """

    def delete(self):
        """
        Called before the source model's .delete() method.
        """

    def get_action(self, name):
        for entry in self.get_action_list():
            if entry.name == name:
                return entry

        raise SourceActionExceptionUnknown(
            'Unknown action `{}` for source `{}`.'.format(
                name, self.label
            )
        )

    def get_action_class_list(self):
        """
        Returns the non initialized list of action classes. This is to allow
        mixins to add their own actions to a base source backend class.
        """
        return self.action_class_list or ()

    def get_action_list(self):
        action_class_list = self.get_action_class_list() or ()

        source = self.get_model_instance()

        for action_class in action_class_list:
            yield action_class(source=source)

    def update(self):
        """
        Called after the source model's .save() method for existing
        instances.
        """


class SourceBackend(
    DynamicFormBackendMixin, ModelBaseBackend,
    SourceBackendMixinSourceMetadata, SourceBackendBase
):
    """
    Final `SourceBackend` class. Separate from `SourceBackendBase` to allow
    for `ModelBaseBackend` to initialize properly while avoiding overriding
    mixin methods like the ones in `SourceBackendMixinSourceMetadata`.
    """
    _backend_app_label = 'sources'
    _backend_model_name = 'Source'
    _loader_module_name = 'source_backends'

    @classmethod
    def get_form_fieldsets(cls):
        fieldsets = (
            (
                _(message='General'), {
                    'fields': ('label', 'enabled')
                }
            ),
        )

        return fieldsets

    @classmethod
    def initialize(cls):
        """
        Optional method for subclasses execute their own initialization
        code.
        """

    @classmethod
    def post_load_modules(cls):
        for source_backend in cls.get_all():
            source_backend.initialize()

    def get_allow_action_execute(self, action, action_execute_kwargs=None):
        return self.get_model_instance().enabled

    def get_upload_form_class(self, action):
        # Hidden import to avoid model that are not ready yet. This happens
        # as `DocumentForm` is imported in `forms.py`.
        from ..forms import UploadBaseForm

        return getattr(self, 'upload_form_class', UploadBaseForm)


class SourceBackendNull(SourceBackend):
    is_visible = False
    label = _(message='Null backend')
