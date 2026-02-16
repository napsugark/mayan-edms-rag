from mayan.apps.testing.tests.base import MayanMigratorTestCase


class Migration0003UniqueFieldsTestCase(MayanMigratorTestCase):
    migrate_from = ('file_metadata', '0002_documenttypesettings')
    migrate_to = ('file_metadata', '0003_auto_20191226_0606')

    def prepare(self):
        StoredDriver = self.old_state.apps.get_model(
            app_label='file_metadata', model_name='StoredDriver'
        )
        StoredDriver.objects.create(
            driver_path='test.path', internal_name='test_internal_name'
        )
        StoredDriver.objects.create(
            driver_path='test.path', internal_name='test_internal_name'
        )

    def test_migration_0003(self):
        StoredDriver = self.new_state.apps.get_model(
            app_label='file_metadata', model_name='StoredDriver'
        )
        self.assertEqual(
            StoredDriver.objects.count(), 1
        )


class Migration0011EntryUniqueInternalNameTestCase(MayanMigratorTestCase):
    migrate_from = ('file_metadata', '0010_add_internal_name')
    migrate_to = ('file_metadata', '0012_add_unique_together')

    def prepare(self):
        StoredDriver = self.old_state.apps.get_model(
            app_label='file_metadata', model_name='StoredDriver'
        )
        DocumentFileDriverEntry = self.old_state.apps.get_model(
            app_label='file_metadata', model_name='DocumentFileDriverEntry'
        )
        FileMetadataEntry = self.old_state.apps.get_model(
            app_label='file_metadata', model_name='FileMetadataEntry'
        )

        Document = self.old_state.apps.get_model(
            app_label='documents', model_name='Document'
        )
        DocumentFile = self.old_state.apps.get_model(
            app_label='documents', model_name='DocumentFile'
        )
        DocumentType = self.old_state.apps.get_model(
            app_label='documents', model_name='DocumentType'
        )

        document_type = DocumentType.objects.create(label='test')
        document = Document.objects.create(document_type=document_type)
        document_file = DocumentFile.objects.create(document=document)

        stored_driver = StoredDriver.objects.create(
            driver_path='test.path', internal_name='test_internal_name'
        )
        document_file_driver_entry = DocumentFileDriverEntry.objects.create(
            document_file=document_file, driver=stored_driver
        )

        FileMetadataEntry.objects.create(
            document_file_driver_entry=document_file_driver_entry,
            key='FileName', value='test'
        )
        FileMetadataEntry.objects.create(
            document_file_driver_entry=document_file_driver_entry,
            key='Filename', value='test'
        )

    def test_migration_0011(self):
        FileMetadataEntry = self.old_state.apps.get_model(
            app_label='file_metadata', model_name='FileMetadataEntry'
        )

        self.assertTrue(
            FileMetadataEntry.objects.get(internal_name='filename')
        )
        self.assertTrue(
            FileMetadataEntry.objects.get(internal_name='filename_1')
        )
