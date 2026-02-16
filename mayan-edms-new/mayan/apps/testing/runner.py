import logging
import shutil

from django import apps
from django.conf import settings
from django.test.runner import DiscoverRunner

from .literals import EXCLUDE_TEST_TAG

logger = logging.getLogger(name=__name__)


class NullMigrationsClass:
    def __contains__(self, item):
        return True

    def __getitem__(self, item):
        return None


class MayanTestRunner(DiscoverRunner):
    @classmethod
    def add_arguments(cls, parser):
        DiscoverRunner.add_arguments(parser)
        parser.add_argument(
            '--mayan-apps', action='store_true', default=False,
            dest='mayan_apps',
            help='Test all Mayan apps that report to have tests.'
        )
        parser.add_argument(
            '--no-exclude', action='store_true', default=False,
            dest='no_exclude',
            help='Include excluded tests.'
        )
        parser.add_argument(
            '--skip-migrations', action='store_true', default=False,
            dest='skip_migrations', help='Skip execution of '
            'migrations after creating the database.'
        )

    def __init__(self, *args, **kwargs):
        self.mayan_apps = kwargs.pop('mayan_apps')
        self.no_exclude = kwargs.pop('no_exclude')
        if kwargs.pop('skip_migrations'):
            settings.MIGRATION_MODULES = NullMigrationsClass()

        super().__init__(*args, **kwargs)
        self.logger = logger

        # Test that should be excluded by default.
        # To include then pass --tag=exclude to the test runner invocation.
        if not self.no_exclude:
            if EXCLUDE_TEST_TAG not in self.tags:
                self.exclude_tags |= {EXCLUDE_TEST_TAG}

    def build_suite(self, *args, **kwargs):
        # Apps that report they have tests.
        if self.mayan_apps:
            args = list(args)
            args[0] = [
                app.name for app in apps.apps.get_app_configs() if getattr(
                    app, 'has_tests', False
                )
            ]

        return super().build_suite(*args, **kwargs)

    def run_tests(self, *args, **kwargs):
        super().run_tests(*args, **kwargs)
        shutil.rmtree(path=settings.MEDIA_ROOT_TEMPORARY)


def filter_tests_by_tags(suite, tags, exclude_tags):
    suite_class = type(suite)
    filtered_suite = suite_class()

    for test in suite:
        if isinstance(test, suite_class):
            filtered_suite.addTests(
                tests=filter_tests_by_tags(
                    suite=test, tags=tags, exclude_tags=exclude_tags
                )
            )
        else:
            test_tags = set(
                getattr(
                    test, 'tags', set()
                )
            )
            test_fn_name = getattr(
                test, '_testMethodName', str(test)
            )
            test_fn = getattr(test, test_fn_name, test)
            test_fn_tags = set(
                getattr(
                    test_fn, 'tags', set()
                )
            )

            upstream_tags = get_test_upstream_tags(test=test)
            all_tags = test_tags.union(test_fn_tags, upstream_tags)
            matched_tags = all_tags.intersection(tags)

            if (matched_tags or not tags) and not all_tags.intersection(exclude_tags):
                filtered_suite.append(test)

    return filtered_suite


def get_test_upstream_tags(test):
    result = set()

    for klass in test.__class__.mro():
        klass_tags = getattr(
            klass, 'tags', set()
        )
        result = result.union(klass_tags)

    return result


def log(self, msg, level=None):
    self.logger = logger

    if level is None:
        level = logging.INFO
    if self.logger is None:
        if self.verbosity <= 0 or (self.verbosity == 1 and level < logging.INFO):
            return
        print(msg)
    else:
        self.logger.log(level, msg)
