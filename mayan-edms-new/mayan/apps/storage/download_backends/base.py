from django.utils.module_loading import import_string

from mayan.apps.mime_types.classes import MIMETypeBackend


class DownloadBackend:
    @classmethod
    def get_backend_class(cls, dotted_path):
        return import_string(dotted_path=dotted_path)

    @classmethod
    def get_backend_instance(cls, dotted_path, arguments=None):
        backend_class = cls.get_backend_class(dotted_path=dotted_path)

        kwargs = arguments or {}

        return backend_class(**kwargs)

    def get_download_filename(self, obj):
        return str(obj)

    def get_download_file_object(self, obj):
        return obj.open(mode='rb')

    def get_download_mime_type_and_encoding(self, obj):
        with obj.open(mode='rb') as file_object:
            mime_type, encoding = MIMETypeBackend.get_backend_instance().get_mime_type(
                file_object=file_object, mime_type_only=True
            )
            return (mime_type, encoding)
