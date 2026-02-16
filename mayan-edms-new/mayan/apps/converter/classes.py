import copy
from io import BytesIO
import logging
import os
import shutil

import PIL
from PIL import Image, ImageFile
import sh

from django.apps import apps
from django.contrib.staticfiles.storage import staticfiles_storage
from django.core.exceptions import ImproperlyConfigured
from django.db import transaction
from django.template import loader
from django.utils.functional import cached_property
from django.utils.module_loading import import_string
from django.utils.translation import gettext_lazy as _

from mayan.apps.mime_types.classes import MIMETypeBackend
from mayan.apps.navigation.links import Link
from mayan.apps.storage.compressed_files import MsgArchive
from mayan.apps.storage.literals import MSG_MIME_TYPES
from mayan.apps.storage.settings import setting_temporary_directory
from mayan.apps.storage.utils import (
    NamedTemporaryFile, TemporaryDirectory, fs_cleanup
)

from .exceptions import (
    AppImageError, InvalidOfficeFormat, LayerError, OfficeConversionError
)
from .literals import (
    CONVERTER_OFFICE_FILE_MIMETYPES, DEFAULT_LIBREOFFICE_PATH,
    DEFAULT_PAGE_NUMBER, DEFAULT_PILLOW_FORMAT, MAP_PILLOW_FORMAT_TO_MIME_TYPE
)
from .literals import IMAGE_ERROR_BROKEN_FILE
from .settings import (
    setting_graphics_backend, setting_graphics_backend_arguments,
    setting_load_truncated_images
)

logger = logging.getLogger(name=__name__)


libreoffice_path = setting_graphics_backend_arguments.value.get(
    'libreoffice_path', DEFAULT_LIBREOFFICE_PATH
)

logger = logging.getLogger(name=__name__)


class AppImageErrorImage:
    _registry = {}

    @classmethod
    def all(cls):
        return cls._registry.values()

    @classmethod
    def get(cls, name):
        return cls._registry[name]

    def __init__(self, name, image_path=None, template_name=None):
        if name in self.__class__._registry:
            raise ImproperlyConfigured(
                '{} already has a entry named `{}`'.format(__class__, name)
            )

        self.name = name
        self.image_path = image_path
        self.template_name = template_name
        self.template = loader.get_template(template_name=self.template_name)

        self.__class__._registry[name] = self

    def open(self):
        return staticfiles_storage.open(name=self.image_path, mode='rb')

    def render(self, context=None):
        return self.template.render(context=context)


class ConverterBase:
    @staticmethod
    def get_converter_class():
        return import_string(dotted_path=setting_graphics_backend.value)

    @staticmethod
    def get_output_content_type():
        output_format = setting_graphics_backend_arguments.value.get(
            'pillow_format', DEFAULT_PILLOW_FORMAT
        )

        return MAP_PILLOW_FORMAT_TO_MIME_TYPE.get(output_format)

    def __init__(self, file_object, mime_type=None):
        ImageFile.LOAD_TRUNCATED_IMAGES = setting_load_truncated_images.value

        self.file_object = file_object
        self.image = None

        self.mime_type = mime_type or MIMETypeBackend.get_backend_instance().get_mime_type(
            file_object=file_object, mime_type_only=False
        )[0]
        self.soffice_file = None
        Image.init()
        try:
            self.command_libreoffice = sh.Command(
                path=libreoffice_path
            ).bake(
                '--headless', '--convert-to', 'pdf:writer_pdf_Export'
            )
        except sh.CommandNotFound:
            self.command_libreoffice = None

    def convert(self, page_number=DEFAULT_PAGE_NUMBER):
        self.page_number = page_number

    def get_page(self, output_format=None):
        output_format = output_format or setting_graphics_backend_arguments.value.get(
            'pillow_format', DEFAULT_PILLOW_FORMAT
        )

        if not self.image:
            self.seek_page(page_number=0)

        image_buffer = BytesIO()
        new_mode = self.image.mode

        if output_format.upper() == 'JPEG':
            # JPEG doesn't support transparency channel, convert the image to
            # RGB. Removes modes: P and RGBA.
            new_mode = 'RGB'

        self.image.convert(
            mode=new_mode
        ).save(fp=image_buffer, format=output_format)

        image_buffer.seek(0)

        return image_buffer

    def get_page_count(self):
        try:
            self.soffice_file = self.to_pdf()
        except InvalidOfficeFormat as exception:
            logger.debug('Is not an office format document; %s', exception)

    def seek_page(self, page_number):
        """
        Seek the specified page number from the source file object.
        If the file is a paged image get the page if not convert it to a
        paged image format and return the specified page as an image.
        """
        # Starting with #0.
        self.file_object.seek(0)

        try:
            self.image = Image.open(fp=self.file_object)
        except IOError:
            # Cannot identify image file.
            self.image = self.convert(page_number=page_number)
        except PIL.Image.DecompressionBombError as exception:
            error_message = 'Unable to seek document page. Increase the '
            'value of the argument "pillow_maximum_image_pixels" in the '
            'CONVERTER_GRAPHICS_BACKEND_ARGUMENTS setting; {}'.format(
                exception
            )
            logger.error(error_message)
            raise AppImageError(
                details=error_message, error_name=IMAGE_ERROR_BROKEN_FILE
            )
        else:
            try:
                self.image.seek(frame=page_number)
            except Exception as exception:
                error_message = 'Unable to seek document page; {}'.format(
                    exception
                )
                raise AppImageError(
                    details=error_message, error_name=IMAGE_ERROR_BROKEN_FILE
                )
            else:
                try:
                    self.image.load()
                except Exception as exception:
                    error_message = 'Unable to load document page; {}'.format(
                        exception
                    )
                    raise AppImageError(
                        details=error_message,
                        error_name=IMAGE_ERROR_BROKEN_FILE
                    )

    def soffice(self):
        """
        Executes LibreOffice as a sub process.
        """
        if not self.command_libreoffice:
            raise OfficeConversionError(
                _(message='LibreOffice not installed or not found.')
            )

        with NamedTemporaryFile() as temporary_file_object:
            # Copy the source file object of the converter instance to a
            # named temporary file to be able to pass it to the LibreOffice
            # execution.
            self.file_object.seek(0)
            shutil.copyfileobj(
                fsrc=self.file_object, fdst=temporary_file_object
            )
            self.file_object.seek(0)
            temporary_file_object.seek(0)

            with TemporaryDirectory() as libreoffice_home_directory:
                args = (
                    temporary_file_object.name, '--outdir', setting_temporary_directory.value,
                    '-env:UserInstallation=file://{}'.format(
                        os.path.join(
                            libreoffice_home_directory,
                            'LibreOffice_Conversion'
                        )
                    ),
                )

                kwargs = {
                    '_env': {'HOME': libreoffice_home_directory}
                }

                if self.mime_type == 'text/plain':
                    kwargs.update(
                        {'infilter': 'Text (encoded):UTF8,LF,,,'}
                    )

                try:
                    self.command_libreoffice(*args, **kwargs)
                except sh.ErrorReturnCode as exception:
                    temporary_file_object.close()
                    raise OfficeConversionError(exception)
                except Exception as exception:
                    temporary_file_object.close()
                    logger.error(
                        'Exception launching LibreOffice; %s', exception,
                        exc_info=True
                    )
                    raise

            # LibreOffice return a PDF file with the same name as the input
            # provided but with the .pdf extension.

            # Get the converted output file path out of the temporary file
            # name plus the temporary directory.

            filename, extension = os.path.splitext(
                os.path.basename(temporary_file_object.name)
            )

            logger.debug('filename: %s', filename)
            logger.debug('extension: %s', extension)

            converted_file_path = os.path.join(
                setting_temporary_directory.value, os.path.extsep.join(
                    (filename, 'pdf')
                )
            )
            logger.debug('converted_file_path: %s', converted_file_path)

        # Don't use context manager with the NamedTemporaryFile on purpose
        # so that it is deleted when the caller closes the file and not
        # before.
        temporary_converted_file_object = NamedTemporaryFile()

        # Copy the LibreOffice output file to a new named temporary file
        # and delete the converted file.
        with open(file=converted_file_path, mode='rb') as converted_file_object:
            shutil.copyfileobj(
                fsrc=converted_file_object,
                fdst=temporary_converted_file_object
            )
        fs_cleanup(filename=converted_file_path)
        temporary_converted_file_object.seek(0)
        return temporary_converted_file_object

    def to_pdf(self):
        # Handle .msg files
        if self.mime_type in MSG_MIME_TYPES:
            archive = MsgArchive.open(file_object=self.file_object)
            members = archive.members()
            if len(members):
                if 'message.txt' in members:
                    self.file_object = archive.open_member(
                        filename='message.txt'
                    )
                else:
                    self.file_object = archive.open_member(
                        filename=members[0]
                    )

                self.mime_type = MIMETypeBackend.get_backend_instance().get_mime_type(
                    file_object=self.file_object, mime_type_only=True
                )[0]

        if self.mime_type in CONVERTER_OFFICE_FILE_MIMETYPES:
            return self.soffice()
        else:
            raise InvalidOfficeFormat(
                _(message='Not an office file format.')
            )

    def transform(self, transformation):
        if not self.image:
            self.seek_page(page_number=0)

        self.image = transformation.execute_on(image=self.image)

    def transform_many(self, transformations):
        if not self.image:
            self.seek_page(page_number=0)

        for transformation in transformations:
            self.image = transformation.execute_on(image=self.image)


class Layer:
    _registry = {}

    @classmethod
    def all(cls):
        return cls._registry.values()

    @classmethod
    def get(cls, name):
        return cls._registry[name]

    @classmethod
    def get_by_value(cls, key, value):
        for name, layer in cls._registry.items():
            if getattr(layer, key) == value:
                return layer

    @classmethod
    def invalidate_cache(cls):
        for layer in cls.all():
            layer.__dict__.pop('stored_layer', None)

    @classmethod
    def update(cls):
        for layer in cls.all():
            layer.stored_layer

    def __init__(
        self, label, name, order, permission_map, default=False,
        empty_results_text=None, icon=None
    ):
        self.default = default
        self.empty_results_text = empty_results_text
        self.label = label
        self.name = name
        self.order = order
        self.permission_map = permission_map
        self.icon = icon

        # Check order
        layer = self.__class__.get_by_value(key='order', value=self.order)

        if layer:
            raise ImproperlyConfigured(
                'Layer "{}" already has order "{}" requested by '
                'layer "{}"'.format(
                    layer.name, order, self.name
                )
            )

        # Check default
        if default:
            layer = self.__class__.get_by_value(key='default', value=True)
            if layer:
                raise ImproperlyConfigured(
                    'Layer "{}" is already the default layer; "{}"'.format(
                        layer.name, self.name
                    )
                )

        self.__class__._registry[name] = self

    def __str__(self):
        return str(self.label)

    def add_transformation_to(
        self, obj, transformation_class, arguments=None, order=None
    ):
        ContentType = apps.get_model(
            app_label='contenttypes', model_name='ContentType'
        )
        content_type = ContentType.objects.get_for_model(model=obj)
        object_layer, created = self.stored_layer.object_layers.get_or_create(
            content_type=content_type, object_id=obj.pk
        )

        if self in transformation_class._layer_transformations:
            return object_layer.transformations.create(
                arguments=arguments or '', order=order,
                name=transformation_class.name
            )
        else:
            raise LayerError(
                'Transformation `{}` not registered for layer `{}`.'.format(
                    transformation_class, self
                )
            )

    def copy_transformations(self, source, targets, delete_existing=False):
        """
        Copy transformation from source to all targets.
        """
        ContentType = apps.get_model(
            app_label='contenttypes', model_name='ContentType'
        )

        transformations = self.get_transformations_for(obj=source)

        with transaction.atomic():
            for target in targets:
                content_type = ContentType.objects.get_for_model(model=target)
                object_layer, created = self.stored_layer.object_layers.get_or_create(
                    content_type=content_type, object_id=target.pk
                )
                if delete_existing:
                    object_layer.transformations.all().delete()

                for transformation in transformations:
                    object_layer.transformations.create(
                        order=transformation.order,
                        name=transformation.name,
                        arguments=transformation.arguments
                    )

    def get_empty_results_text(self):
        if self.empty_results_text:
            return self.empty_results_text
        else:
            return _(
                message='Transformations allow changing the visual appearance '
                'of documents without making permanent changes to the '
                'document file themselves.'
            )

    def get_model_instance(self):
        StoredLayer = apps.get_model(
            app_label='converter', model_name='StoredLayer'
        )
        stored_layer, created = StoredLayer.objects.update_or_create(
            name=self.name, defaults={'order': self.order}
        )

        return stored_layer

    def get_permission(self, action):
        return self.permission_map.get(action, None)

    def get_transformations_for(self, obj, as_classes=False):
        """
        as_classes == True returns the transformation classes from .classes
        ready to be feed to the converter class.
        """
        LayerTransformation = apps.get_model(
            app_label='converter', model_name='LayerTransformation'
        )

        return LayerTransformation.objects.get_for_object(
            as_classes=as_classes, obj=obj,
            only_stored_layer=self.stored_layer
        )

    @cached_property
    def stored_layer(self):
        return self.get_model_instance()


class LayerLink(Link):
    def __init__(self, action, layer=None, multi_item=False, **kwargs):
        super().__init__(**kwargs)
        self.action = action
        self.layer = layer
        self.multi_item = multi_item

    def __repr__(self):
        return '<LayerLink ({} | {})>'.format(
            self.layer, self.action
        )

    def copy(self, layer):
        result = copy.copy(self)
        result.layer = layer

        return result

    def get_icon(self, context):
        if self.action == 'view':
            layer = self.get_layer(context=context)
            if layer and layer.icon:
                return layer.icon

        return super().get_icon(context=context)

    def get_kwargs(self, context):
        ContentType = apps.get_model(
            app_label='contenttypes', model_name='ContentType'
        )

        if 'content_object' in context:
            content_object_variable = 'content_object'
        else:
            content_object_variable = 'resolved_object'

        content_type = ContentType.objects.get_for_model(
            context[content_object_variable]
        )
        layer = self.get_layer(context=context)

        layer_name = layer.name

        result = {
            'app_label': '"{}"'.format(content_type.app_label),
            'model_name': '"{}"'.format(content_type.model),
            'object_id': '{}.pk'.format(content_object_variable),
            'layer_name': '"{}"'.format(layer_name)
        }

        if self.action in ('delete', 'edit') and not self.multi_item:
            result['transformation_id'] = 'object.pk'

        return result

    def get_layer(self, context=None):
        context = context or {}

        if self.layer:
            return self.layer
        elif 'layer' in context:
            return context['layer']
        else:
            layer_name = context.get('layer_name', None)
            if layer_name:
                return Layer.get(name=layer_name)
            else:
                return Layer.get_by_value(key='default', value=True)

    def get_permission_object(self, context):
        try:
            return context['content_object']
        except KeyError:
            try:
                return context['resolved_object']
            except KeyError:
                return None

    def get_permission(self, context):
        layer = self.get_layer(context=context)
        permission = layer.get_permission(action=self.action)

        if permission:
            return permission
        else:
            return None
