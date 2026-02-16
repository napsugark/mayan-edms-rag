import logging

from django.core.files.base import File

from mayan.apps.mime_types.classes import MIMETypeBackend
from mayan.apps.rest_api import generics
from mayan.apps.rest_api.api_view_mixins import (
    ExternalContentTypeObjectAPIViewMixin
)

from .api_view_mixins import APIImageViewMixin
from .classes import AppImageErrorImage
from .models import Asset
from .permissions import (
    permission_asset_create, permission_asset_delete, permission_asset_edit,
    permission_asset_view
)
from .serializers import AppImageErrorSerializer, AssetSerializer

logger = logging.getLogger(name=__name__)


class APIAssetListView(generics.ListCreateAPIView):
    """
    get: Returns a list of all the assets.
    post: Create a new asset.
    """
    mayan_object_permission_map = {'GET': permission_asset_view}
    mayan_view_permission_map = {'POST': permission_asset_create}
    serializer_class = AssetSerializer
    source_queryset = Asset.objects.all()

    def get_instance_extra_data(self):
        return {'_event_actor': self.request.user}


class APIAssetDetailView(generics.RetrieveUpdateDestroyAPIView):
    """
    delete: Delete the selected asset.
    get: Return the details of the selected asset.
    patch: Edit the properties of the selected asset.
    put: Edit the properties of the selected asset.
    """
    lookup_url_kwarg = 'asset_id'
    mayan_object_permission_map = {
        'DELETE': permission_asset_delete,
        'GET': permission_asset_view,
        'PATCH': permission_asset_edit,
        'PUT': permission_asset_edit
    }
    serializer_class = AssetSerializer
    source_queryset = Asset.objects.all()

    def get_instance_extra_data(self):
        return {'_event_actor': self.request.user}


class APIAssetImageView(APIImageViewMixin, generics.RetrieveAPIView):
    """
    get: Returns an image representation of the selected asset.
    """
    lookup_url_kwarg = 'asset_id'
    mayan_object_permission_map = {'GET': permission_asset_view}
    source_queryset = Asset.objects.all()


class APIAppImageErrorDetailView(generics.RetrieveAPIView):
    """
    get: Returns the details of the selected app image error.
    """
    lookup_url_kwarg = 'app_image_error_name'
    serializer_class = AppImageErrorSerializer

    def get_object(self):
        return AppImageErrorImage.get(
            name=self.kwargs[self.lookup_url_kwarg]
        )


class APIAppImageErrorImageView(APIImageViewMixin, generics.RetrieveAPIView):
    """
    get: Returns an image representation of the selected app image error.
    """
    lookup_url_kwarg = 'app_image_error_name'

    def get_file_generator(self):
        def file_generator():
            with self.get_object().open() as file_object:
                while True:
                    chunk = file_object.read(File.DEFAULT_CHUNK_SIZE)
                    if not chunk:
                        break
                    else:
                        yield chunk

        return file_generator

    def get_object(self):
        return AppImageErrorImage.get(
            name=self.kwargs[self.lookup_url_kwarg]
        )

    def get_stream_mime_type(self):
        mime_type_backend = MIMETypeBackend.get_backend_instance()
        with self.get_object().open() as file_object:
            mime_type, mime_encoding = mime_type_backend.get_mime_type(
                file_object=file_object, mime_type_only=True
            )
            return mime_type

    def set_cache_file(self, request):
        return


class APIAppImageErrorListView(generics.ListAPIView):
    """
    get: Returns an list of the available app image errors.
    """
    serializer_class = AppImageErrorSerializer

    def get_source_queryset(self):
        return list(
            AppImageErrorImage.all()
        )


class APIContentObjectImageView(
    APIImageViewMixin, ExternalContentTypeObjectAPIViewMixin,
    generics.RetrieveAPIView
):
    """
    get: Returns an image representation of the selected content object.
    """
    def get_content_type(self):
        return self.content_type

    def set_object(self):
        self.obj = self.get_external_object()
