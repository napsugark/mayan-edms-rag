import logging

from django.apps import apps
from django.conf import settings
from django.db.utils import IntegrityError, OperationalError, ProgrammingError
from django.utils.functional import classproperty
from django.utils.translation import gettext_lazy as _

from mayan.apps.common.class_mixins import AppsModuleLoaderMixin
from mayan.apps.common.utils import (
    convert_to_internal_name, deduplicate_dictionary_values,
    get_class_full_name
)
from mayan.apps.templating.template_backends import Template

from .exceptions import FileMetadataError
from .literals import ERROR_LOG_DOMAIN_NAME
from .settings import setting_auto_process, setting_drivers_arguments

logger = logging.getLogger(name=__name__)


class FileMetadataDriverCollection:
    _driver_to_mime_type_dict = {}
    _mime_type_to_driver_dict = {}

    @classmethod
    def do_driver_register(cls, klass):
        if klass in cls._driver_to_mime_type_dict:
            raise FileMetadataError(
                'Driver "{}" is already registered.'.format(klass)
            )

        cls._driver_to_mime_type_dict[klass] = klass.mime_type_list

        for mime_type in klass.mime_type_list:
            cls._mime_type_to_driver_dict.setdefault(
                mime_type, []
            ).append(klass)

        klass.dotted_path = get_class_full_name(klass=klass)

    @classmethod
    def get_all(cls, sorted=False):
        result = list(
            cls._driver_to_mime_type_dict.keys()
        )
        if sorted:
            result.sort(key=lambda driver: driver.label)

        return result

    @classmethod
    def get_driver_for_mime_type(cls, mime_type):
        driver_class_list = cls._mime_type_to_driver_dict.get(
            mime_type, ()
        )
        # Add wildcard drivers, drivers meant to be executed for all MIME
        # types.
        driver_class_list += tuple(
            cls._mime_type_to_driver_dict.get(
                '*', ()
            )
        )
        driver_class_list = [
            driver_class for driver_class in driver_class_list if driver_class.model_instance
        ]

        return driver_class_list


class FileMetadataDriverMetaclass(type):
    def __new__(mcs, name, bases, attrs):
        new_class = super().__new__(mcs, name, bases, attrs)

        if not new_class.__module__ == __name__:
            FileMetadataDriverCollection.do_driver_register(klass=new_class)

        return new_class


class FileMetadataDriver(
    AppsModuleLoaderMixin, metaclass=FileMetadataDriverMetaclass
):
    _loader_module_name = 'drivers'
    argument_name_list = ()
    description = ''
    # Workaround a Django bug that causes the template system to call the
    # class which would cause it to create an instance without required
    # arguments and lead to an empty entry in list views.
    # https://stackoverflow.com/questions/6861601/cannot-resolve-callable-context-variable
    do_not_call_in_templates = True
    dotted_path_previous_list = ()
    enabled = None
    """
    Default value the `enabled` field when driver is populated or new
    document types are added.
    `None` the final value will be up to the setting `setting_auto_process`.
    `True` will be enabled.
    `False` will not be enabled.
    Don't access the property directly, use cls.get_enabled_value().
    """
    internal_name = None
    label = None
    mime_type_list = ()
    model_instance = None

    @classproperty
    def collection(cls):
        if cls != FileMetadataDriver:
            raise AttributeError(
                'This method is only available to the parent class.'
            )
        return FileMetadataDriverCollection

    @classmethod
    def do_document_file_process(cls, document_file):
        arguments = cls.get_argument_values_for_document_file(
            document_file=document_file
        )

        driver_instance = cls(**arguments)

        driver_instance._instance_do_document_file_process(
            document_file=document_file
        )

    @classmethod
    def do_model_instance_populate(cls):
        DocumentType = apps.get_model(
            app_label='documents', model_name='DocumentType'
        )
        DocumentTypeDriverConfiguration = apps.get_model(
            app_label='file_metadata',
            model_name='DocumentTypeDriverConfiguration'
        )
        StoredDriver = apps.get_model(
            app_label='file_metadata', model_name='StoredDriver'
        )

        try:
            model_instance, created = StoredDriver.objects.update_or_create(
                driver_path=cls.dotted_path, defaults={
                    'exists': True, 'internal_name': cls.internal_name
                }
            )
        except IntegrityError:
            # May be a driver that moved to another location.
            model_instance = StoredDriver.objects.get(
                internal_name=cls.internal_name
            )

            if model_instance.driver_path in cls.dotted_path_previous_list:
                model_instance.driver_path = cls.dotted_path
                model_instance.exists = True
                model_instance.save()

                created = False
            else:
                # Unknown situation, re-raise original error.
                raise

        cls.model_instance = model_instance

        if created:
            enabled = cls.get_enabled_value()
            for document_type in DocumentType.objects.all():
                DocumentTypeDriverConfiguration.objects.update_or_create(
                    defaults={'enabled': enabled},
                    document_type=document_type, stored_driver=model_instance
                )

    @classmethod
    def get_argument_name_list(cls):
        return cls.argument_name_list

    @classmethod
    def get_argument_values_for_document_file(cls, document_file):
        document_type = document_file.document.document_type

        configuration_instance = document_type.file_metadata_driver_configurations.get(
            stored_driver__internal_name=cls.internal_name
        )

        document_type_arguments = configuration_instance.get_arguments()

        context = {'document_file': document_file}

        argument_values_rendered = {}

        for key, value in document_type_arguments.items():
            template = Template(template_string=value)
            template_result = template.render(context=context)
            argument_values_rendered[key] = template_result

        return argument_values_rendered

    @classmethod
    def get_argument_values_from_settings(cls):
        result = {}

        raw_argument_values = setting_drivers_arguments.value.get(
            cls.internal_name, {}
        )

        argument_name_list = cls.get_argument_name_list()

        for argument_name in argument_name_list:
            value = raw_argument_values.get(argument_name)
            if value:
                result[argument_name] = value

        return result

    @classmethod
    def get_argument_values_from_settings_display(cls):
        try:
            arguments = cls.get_argument_values_from_settings()
        except Exception as exception:
            arguments = {
                '__error__': _(
                    message='Badly formatted arguments YAML; %s'
                ) % exception
            }

        result = []

        argument_name_list = cls.get_argument_name_list()

        for argument_name in argument_name_list:
            value = arguments.get(argument_name)
            result.append(
                '{}: {}'.format(argument_name, value)
            )

        return ', '.join(result)

    @classmethod
    def get_enabled_value_from_settings(cls):
        raw_argument_values = setting_drivers_arguments.value.get(
            cls.internal_name, {}
        )

        return raw_argument_values.get('enabled', None)

    @classmethod
    def get_enabled_value(cls):
        """
        Calculate the final value of the class enabled field when populated
        or for new document types.

        - `None` the final value will be up to the setting
          `setting_auto_process`.
        - `True` will be enabled.
        - `False` will not be enabled.

        The value itself an be overridden by the setting
        `FILE_METADATA_DRIVERS_ARGUMENTS` to change the behavior per driver.
        """

        enabled_from_settings = cls.get_enabled_value_from_settings()

        if enabled_from_settings is None:
            if cls.enabled is None:
                return setting_auto_process.value
            else:
                return cls.enabled
        else:
            return enabled_from_settings

    @classmethod
    def get_form_class(cls):
        return getattr(cls, 'Form', None)

    @classmethod
    def get_mime_type_list_display(cls):
        return ', '.join(cls.mime_type_list)

    @classmethod
    def initialize(cls):
        StoredDriver = apps.get_model(
            app_label='file_metadata', model_name='StoredDriver'
        )

        try:
            """
            Check is the table is ready.
            If not, this will log an error similar to this:
            2023-12-12 09:12:46.985 UTC [79] ERROR:  relation "file_metadata_storeddriver" does not exist at character 22
            2023-12-12 09:12:46.985 UTC [79] STATEMENT:  SELECT 1 AS "a" FROM "file_metadata_storeddriver" LIMIT 1
            This error is expected and should be ignored.
            """
            StoredDriver.objects.exists()
        except (OperationalError, ProgrammingError):
            """
            This error is expected when trying to initialize the
            stored permissions during the initial creation of the
            database. Can be safely ignored under that situation.
            """
        else:
            try:
                # Reset all `StoredDriver` in case a file metadata app was
                # disabled.
                StoredDriver.objects.update(exists=False)
            except (OperationalError, ProgrammingError):
                """
                This error is expected when performing an upgrade between
                series 4.7 and 4.8 as the fields `exists` has not yet been
                created by the migration.
                """
            else:
                for driver in cls.collection.get_all():
                    driver.do_model_instance_populate()

    def _instance_do_document_file_process(self, document_file):
        try:
            logger.debug(
                'Starting processing document file: %s', document_file
            )

            FileMetadataEntry = apps.get_model(
                app_label='file_metadata', model_name='FileMetadataEntry'
            )

            file_metadata_dictionary = self._process(
                document_file=document_file
            )

            file_metadata_dictionary = file_metadata_dictionary or {}

            internal_name_dictionary = {}
            for key in file_metadata_dictionary.keys():
                internal_name_dictionary[key] = convert_to_internal_name(
                    value=key
                )

            internal_name_dictionary_deduplicated = deduplicate_dictionary_values(
                dictionary=internal_name_dictionary
            )

            queryset_document_file_metadata = self.model_instance.driver_entries.filter(
                document_file=document_file
            )
            queryset_document_file_metadata.delete()

            document_file_driver_entry = self.model_instance.driver_entries.create(
                document_file=document_file
            )

            coroutine = FileMetadataEntry.objects.create_bulk()
            next(coroutine)

            for key, value in file_metadata_dictionary.items():
                internal_name = internal_name_dictionary_deduplicated[key]

                # Drivers should not be returning `None` values.
                # Added to workaround undocumented backward incompatible
                # changes in Ollama.
                value_clean = value or ''

                coroutine.send(
                    {
                        'document_file_driver_entry': document_file_driver_entry,
                        'internal_name': internal_name, 'key': key,
                        'value': value_clean
                    }
                )

            coroutine.close()
        except Exception as exception:
            if settings.DEBUG and 0:
                raise
            else:
                error_log_text = '''
                Cannot process driver `{driver}` for document file `{document_file}`; {exception}
                '''.format(
                    document_file=document_file, driver=self.label,
                    exception=exception
                )
                document_file.error_log.create(
                    domain_name=ERROR_LOG_DOMAIN_NAME, text=error_log_text
                )

    def _process(self, document_file):
        raise NotImplementedError(
            'Your %s class has not defined the required _process() '
            'method.' % self.__class__.__name__
        )
