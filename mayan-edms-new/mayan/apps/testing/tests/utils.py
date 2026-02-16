from contextlib import contextmanager
import sys


class NullFile:
    def flush(self):
        """Has no effect."""

    def write(self, string):
        """Writes here go nowhere."""


def as_id_list(items):
    return ','.join(
        [
            str(item.pk) for item in items
        ]
    )


@contextmanager
def mute_stdout():
    stdout_old = sys.stdout
    sys.stdout = NullFile()
    yield
    sys.stdout = stdout_old
