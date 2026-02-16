from .base import DownloadBackend


class DownloadBackendFileDirectStorage(DownloadBackend):
    def get_download_file_object(self, obj):
        return obj.open(mode='rb')
