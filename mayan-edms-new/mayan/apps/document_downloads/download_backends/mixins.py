class DownloadBackendMixinDocumentFile:
    def get_stored_filename(self, obj):
        return obj.file.name
