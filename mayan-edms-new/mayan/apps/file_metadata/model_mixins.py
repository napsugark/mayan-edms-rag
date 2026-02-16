from django.utils.functional import cached_property
from django.utils.module_loading import import_string
from django.utils.translation import gettext_lazy as _

from mayan.apps.common.serialization import yaml_load


class DocumentFileDriverEntryBusinessLogicMixin:
    def get_attribute_count(self):
        return self.entries.count()
    get_attribute_count.short_description = _(message='Attribute count')


class DocumentTypeDriverConfigurationBusinessLogicMixin:
    def get_arguments(self):
        driver_class = self.stored_driver.driver_class

        argument_values_from_settings = driver_class.get_argument_values_from_settings()

        stored_arguments = yaml_load(stream=self.arguments or '{}')

        for argument_name in self.stored_driver.driver_class.get_argument_name_list():
            value = stored_arguments.get(argument_name)

            if value:
                argument_values_from_settings[argument_name] = value

        return argument_values_from_settings

    def get_arguments_display(self):
        try:
            arguments = self.get_arguments()
        except Exception as exception:
            arguments = {
                'error': _(
                    message='Badly formatted arguments YAML; %s'
                ) % exception
            }

        result = []
        for key, value in arguments.items():
            result.append(
                '{}: {}'.format(key, value)
            )

        return ', '.join(result)

    get_arguments_display.short_description = _(message='Arguments')


class FileMetadataEntryBusinessLogicMixin:
    def get_full_path(self):
        return '{}__{}'.format(
            self.document_file_driver_entry.driver.internal_name,
            self.internal_name
        )

    get_full_path.short_description = _(message='Full path')
    get_full_path.help_text = _(
        message='Path used to access the value of the file metadata entry.'
    )


class StoredDriverBusinessLogicMixin:
    @cached_property
    def driver_class(self):
        return import_string(dotted_path=self.driver_path)

    @cached_property
    def driver_label(self):
        return self.driver_class.label
