TEST_DRIVER_CLASS_PATH = 'mayan.apps.file_metadata.tests.drivers.FileMetadataDriverTest'
TEST_DRIVER_INTERNAL_NAME = 'test_driver'
TEST_DRIVER_LABEL = 'test label'

TEST_FILE_METADATA_KEY = 'test_key'
TEST_FILE_METADATA_VALUE = 'test_value'

TEST_FILE_METADATA_INDEX_NODE_TEMPLATE = "{{{{ document.file_metadata_value_of.{}__{} }}}}".format(
    TEST_DRIVER_INTERNAL_NAME, TEST_FILE_METADATA_KEY
)
