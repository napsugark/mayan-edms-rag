import shutil

from django.utils.encoding import force_str
from django.utils.module_loading import import_string

from mayan.apps.converter.classes import ConverterBase
from mayan.apps.storage.utils import TemporaryFile

from .settings import setting_ocr_backend, setting_ocr_backend_arguments


class OCRBackendBase:
    @staticmethod
    def get_instance():
        return import_string(
            dotted_path=setting_ocr_backend.value
        )(**setting_ocr_backend_arguments.value)

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def execute(self, file_object, language=None, transformations=None):
        self.language = language

        if not transformations:
            transformations = []

        self.converter = ConverterBase.get_converter_class()(
            file_object=file_object
        )

        for transformation in transformations:
            self.converter.transform(transformation=transformation)

        image = self.converter.get_page()

        with TemporaryFile() as temporary_image_file:
            shutil.copyfileobj(fsrc=image, fdst=temporary_image_file)
            temporary_image_file.seek(0)

            return force_str(
                s=self._execute(image_file_object=temporary_image_file)
            )
