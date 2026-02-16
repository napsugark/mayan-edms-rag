from mayan.apps.app_manager.apps import MayanAppConfig


class MOTDApp(MayanAppConfig):
    has_tests = False
    has_app_translations = False
    has_javascript_translations = False
    name = 'mayan.apps.motd'
