from mayan.apps.testing.tests.base import MayanMigratorTestCase

from ..source_backends import SourceBackendWebForm


class SourceBackendPathMigrationTestCase(MayanMigratorTestCase):
    migrate_from = ('sources', '0028_auto_20210905_0558')
    migrate_to = ('source_web_forms', '0003_fix_backend_paths')

    def prepare(self):
        Source = self.old_state.apps.get_model(
            app_label='sources', model_name='Source'
        )

        Source.objects.create(
            backend_path='mayan.apps.sources.source_backends.web_form_backends.SourceBackendWebForm',
            label='test source web form'
        )

    def test_source_backend_path_updates(self):
        Source = self.old_state.apps.get_model(
            app_label='sources', model_name='Source'
        )

        self.assertEqual(
            Source.objects.get(label='test source web form').backend_path,
            SourceBackendWebForm.get_class_path()
        )
