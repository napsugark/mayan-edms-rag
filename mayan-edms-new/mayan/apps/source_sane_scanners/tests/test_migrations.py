from mayan.apps.testing.tests.base import MayanMigratorTestCase

from ..source_backends import SourceBackendSANEScanner


class SourceBackendPathMigrationTestCase(MayanMigratorTestCase):
    migrate_from = ('sources', '0028_auto_20210905_0558')
    migrate_to = ('source_sane_scanners', '0003_fix_backend_paths')

    def prepare(self):
        Source = self.old_state.apps.get_model(
            app_label='sources', model_name='Source'
        )

        Source.objects.create(
            backend_path='mayan.apps.sources.source_backends.sane_scanner_backends.SourceBackendSANEScanner',
            label='test source SANE'
        )

    def test_source_backend_path_updates(self):
        Source = self.old_state.apps.get_model(
            app_label='sources', model_name='Source'
        )

        self.assertEqual(
            Source.objects.get(label='test source SANE').backend_path,
            SourceBackendSANEScanner.get_class_path()
        )
