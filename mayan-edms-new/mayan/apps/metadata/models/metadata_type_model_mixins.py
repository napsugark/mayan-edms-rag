from mayan.apps.common.utils import comma_splitter
from mayan.apps.templating.template_backends import Template


class MetadataTypeBusinessLogicMixin:
    def get_default_value(self):
        template = Template(template_string=self.default)
        return template.render()

    def get_lookup_values(self):
        template = Template(
            context_entry_name_list=('groups', 'users'),
            template_string=self.lookup
        )

        template_result = template.render()

        return comma_splitter(string=template_result)

    def get_required_for(self, document_type):
        """
        Determine if the metadata type is required for the specified document
        type.
        """
        queryset = document_type.metadata.filter(
            required=True, metadata_type=self
        )

        return queryset.exists()
