import os

from django.conf import settings

TEST_DEPENDENCY_GROUP_NAME = 'environment'
TEST_DEPENDENCY_GROUP_ENTRY_NAME = 'testing'

TEST_TAR_CVE_2007_4559_FILENAME = 'CVE_2007_4559.tar'

TEST_TAR_CVE_2007_4559_PATH = os.path.join(
    settings.BASE_DIR, 'apps', 'dependencies', 'tests', 'contrib',
    'test_files', TEST_TAR_CVE_2007_4559_FILENAME
)


MESSAGE_TEST_MORE_RECENT = 'more recent'
MESSAGE_TEST_NOT_LATEST = 'is outdated'
MESSAGE_TEST_UNKNOWN_VERSION = 'not possible'
MESSAGE_TEST_UP_TO_DATE = 'up-to-date'
