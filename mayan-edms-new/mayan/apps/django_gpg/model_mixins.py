from datetime import datetime
from io import BytesIO
import logging

from django.utils.encoding import force_bytes, force_str
from django.utils.timezone import make_aware

from .classes import GPGBackend
from .exceptions import NeedPassphrase, PassphraseError
from .literals import (
    ERROR_MSG_BAD_PASSPHRASE, ERROR_MSG_GOOD_PASSPHRASE,
    ERROR_MSG_MISSING_PASSPHRASE, KEY_TYPE_SECRET,
    OUTPUT_MESSAGE_CONTAINS_PRIVATE_KEY
)

logger = logging.getLogger(name=__name__)


class KeyBusinessLogicMixin:
    @property
    def key_id(self):
        """
        Short form key ID (using the first 8 characters).
        """
        return self.fingerprint[-8:]

    def introspect_key_data(self):
        # Fix the encoding of the key data stream.
        self.key_data = force_str(s=self.key_data)
        import_results, key_info = GPGBackend.get_instance().import_and_list_keys(
            key_data=self.key_data
        )
        logger.debug('key_info: %s', key_info)

        self.algorithm = key_info['algo']
        self.creation_date = make_aware(
            value=datetime.fromtimestamp(
                int(
                    key_info['date']
                )
            )
        )
        if key_info['expires']:
            self.expiration_date = make_aware(
                value=datetime.fromtimestamp(
                    int(
                        key_info['expires']
                    )
                )
            )
        self.fingerprint = key_info['fingerprint']
        self.length = int(
            key_info['length']
        )
        self.user_id = key_info['uids'][0]
        if OUTPUT_MESSAGE_CONTAINS_PRIVATE_KEY in import_results.results[0]['text']:
            self.key_type = KEY_TYPE_SECRET
        else:
            self.key_type = key_info['type']

    def open(self, **kwargs):
        # Discard any keyword arguments. Key content is always served in the
        # same format.
        output_buffer = BytesIO(
            initial_bytes=force_bytes(s=self.key_data)
        )
        return output_buffer

    def sign_file(
        self, file_object, binary=False, clearsign=False, detached=False,
        output=None, passphrase=None
    ):
        """
        Digitally sign a file
        WARNING: using clearsign=True and subsequent decryption corrupts the
        file. Appears to be a problem in python-gnupg or gpg itself.
        https://bugs.debian.org/cgi-bin/bugreport.cgi?bug=55647
        "The problems differ from run to run and file to
        file, and appear to be due to random data being inserted in the
        output data stream."
        """
        file_sign_results = GPGBackend.get_instance().sign_file(
            binary=binary, clearsign=clearsign, detached=detached,
            file_object=file_object, key_data=self.key_data,
            output=output, passphrase=passphrase
        )

        logger.debug(
            'file_sign_results.stderr: %s', file_sign_results.stderr
        )

        if ERROR_MSG_MISSING_PASSPHRASE in file_sign_results.stderr:
            if ERROR_MSG_GOOD_PASSPHRASE not in file_sign_results.stderr:
                raise NeedPassphrase

        if ERROR_MSG_BAD_PASSPHRASE in file_sign_results.stderr:
            raise PassphraseError

        return file_sign_results
