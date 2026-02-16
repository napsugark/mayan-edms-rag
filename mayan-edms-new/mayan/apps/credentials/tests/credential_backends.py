from ..classes import CredentialBackend


class TestCredentialBackend(CredentialBackend):
    label = 'Test source backend'

    @classmethod
    def get_form_fields(cls):
        fields = super().get_form_fields()

        fields.update(
            {
                'test_field': {
                    'label': 'Test field',
                    'class': 'django.forms.CharField',
                    'required': False
                }
            }
        )

        return fields

    @classmethod
    def get_form_fieldsets(cls):
        fieldsets = super().get_form_fieldsets()

        fieldsets += (
            (
                'Testing', {
                    'fields': ('test_field')
                },
            ),
        )

        return fieldsets

    def process_documents(self, dry_run=False):
        """Do nothing. This method is added to allow view testing."""


class TestCredentialBackendWithPostProcessing(TestCredentialBackend):
    label = 'Test source backend with post processing'

    @classmethod
    def post_processing(cls, obj):
        return obj
