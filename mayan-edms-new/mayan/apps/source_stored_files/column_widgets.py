from mayan.apps.documents.column_widgets import ThumbnailWidget


class StoredFileThumbnailWidget(ThumbnailWidget):
    gallery_name = 'sources:stored_file_list'

    def disable_condition(self):
        return True
