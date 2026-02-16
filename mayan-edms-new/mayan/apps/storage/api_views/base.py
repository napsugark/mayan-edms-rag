from mayan.apps.rest_api import generics

from ..views.mixins import ViewMixinDownload, ViewMixinDownloadEvent


class APIObjectDownloadView(
    ViewMixinDownloadEvent, ViewMixinDownload, generics.RetrieveAPIView
):
    def get_download_event_target(self):
        return self.get_object()

    def get_download_file_object(self):
        instance = self.get_object()
        return instance.open(mode='rb')

    def get_serializer(self, *args, **kwargs):
        return None

    def get_serializer_class(self):
        return None

    def retrieve(self, request, *args, **kwargs):
        return self.render_to_response()


class APIObjectDownloadBackendDownloadView(
    ViewMixinDownloadEvent, ViewMixinDownload, generics.RetrieveAPIView
):
    def get_download_event_target(self):
        return self.get_object()

    def get_download_file_object(self):
        instance = self.get_object()
        return instance.open(mode='rb')

    def get_serializer(self, *args, **kwargs):
        return None

    def get_serializer_class(self):
        return None

    def retrieve(self, request, *args, **kwargs):
        return self.render_to_response()
