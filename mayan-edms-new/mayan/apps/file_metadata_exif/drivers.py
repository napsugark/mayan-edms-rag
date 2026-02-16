import json
import logging
from pathlib import Path

import sh

from django.utils.translation import gettext_lazy as _

from mayan.apps.file_metadata.classes import FileMetadataDriver
from mayan.apps.storage.utils import TemporaryDirectory

from .literals import DEFAULT_EXIF_PATH

logger = logging.getLogger(name=__name__)


class FileMetadataDriverEXIF(FileMetadataDriver):
    argument_name_list = ('exiftool_path',)
    description = _(message='Read meta information stored in files.')
    dotted_path_previous_list = (
        'mayan.apps.file_metadata.drivers.exiftool.EXIFToolDriver',
    )
    internal_name = 'exiftool'
    label = _(message='EXIF Tool')
    mime_type_list = ('*',)

    @classmethod
    def get_argument_values_from_settings(cls):
        result = {'exiftool_path': DEFAULT_EXIF_PATH}

        setting_arguments = super().get_argument_values_from_settings()

        if setting_arguments:
            result.update(setting_arguments)

        return result

    def __init__(self, exiftool_path, **kwargs):
        super().__init__(**kwargs)

        self.command_exiftool = sh.Command(path=exiftool_path)
        self.command_exiftool = self.command_exiftool.bake('-j')

    def _process(self, document_file):
        if self.command_exiftool:
            with TemporaryDirectory() as temporary_folder:
                path_temporary_file = Path(
                    temporary_folder, Path(document_file.filename).name
                )

                with path_temporary_file.open(mode='xb') as temporary_fileobject:
                    document_file.save_to_file(
                        file_object=temporary_fileobject
                    )
                    temporary_fileobject.seek(0)
                    try:
                        output = self.command_exiftool(
                            temporary_fileobject.name
                        )
                    except sh.ErrorReturnCode_1 as exception:
                        result = json.loads(s=exception.stdout)[0]
                        if result.get('Error', '') == 'Unknown file type':
                            # Not a fatal error.
                            return result
                    else:
                        return json.loads(s=output)[0]
        else:
            logger.error(
                'EXIFTool binary not found, not processing document '
                'file: %s', document_file
            )
