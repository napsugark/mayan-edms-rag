from mayan.apps.common.classes import PropertyHelper


class DocumentFileMetadataHelper(PropertyHelper):
    @staticmethod
    @property
    def constructor(*args, **kwargs):
        return DocumentFileMetadataHelper(*args, **kwargs)

    def get_result(self, name):
        file_latest = self.instance.file_latest

        if file_latest:
            return file_latest.get_file_metadata(dotted_name=name)
        else:
            return None


class DocumentFileFileMetadataHelper(PropertyHelper):
    @staticmethod
    @property
    def constructor(*args, **kwargs):
        return DocumentFileFileMetadataHelper(*args, **kwargs)

    def get_result(self, name):
        result = self.instance.get_file_metadata(dotted_name=name)
        return result
