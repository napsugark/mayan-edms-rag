from django.apps import apps
from django.core import checks
from django.test.runner import DiscoverRunner

from mayan.apps.app_manager.apps import MayanAppConfig


def check_app_tests(app_configs=None, **kwargs):
    error_list = []

    runner = DiscoverRunner()

    if app_configs is None:
        app_configs = apps.get_app_configs()

    for app_config in app_configs:
        if issubclass(app_config.__class__, MayanAppConfig):
            has_tests = getattr(app_config, 'has_tests', False)

            test_suite = runner.build_suite(
                test_labels=(app_config.module.__name__,)
            )
            test_suite_test_count = test_suite.countTestCases()

            if has_tests and not test_suite_test_count:
                error_list.append(
                    checks.Warning(
                        hint='Ensure app really defines tests, that the '
                        '`has_tests` attribute is correct, and/or that the '
                        'tests load correctly.',
                        id='mayan_apps.W001',
                        msg='App specifies it has tests but none was found.',
                        obj=app_config
                    )
                )

            if not has_tests and test_suite_test_count:
                error_list.append(
                    checks.Warning(
                        hint='Ensure app really does not define tests, that '
                        'the `has_tests` attribute is correct.',
                        id='mayan_apps.W002',
                        msg='App specifies it does not has tests but some '
                        'were found.',
                        obj=app_config
                    )
                )

    return error_list
