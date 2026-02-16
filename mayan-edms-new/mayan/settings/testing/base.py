from tempfile import mkdtemp

from .. import *  # NOQA

MEDIA_ROOT_TEMPORARY = mkdtemp()
MEDIA_ROOT = MEDIA_ROOT_TEMPORARY
setting_namespace.get_setting(name='MEDIA_ROOT').set_value(
    value=MEDIA_ROOT_TEMPORARY
)
setting_namespace.update_globals(
    global_symbol_table=globals()
)

AUTHENTICATION_BACKEND = 'mayan.apps.authentication.authentication_backends.AuthenticationBackendModelDjangoDefault'

CELERY_BROKER_URL = 'memory://'
CELERY_TASK_ALWAYS_EAGER = True
CELERY_TASK_EAGER_PROPAGATES = True

COMMON_HOME_VIEW_DASHBOARD_NAME = None

DOCUMENT_PARSING_AUTO_PARSING = False

FILE_METADATA_AUTO_PROCESS = False

INSTALLED_APPS = [
    cls for cls in INSTALLED_APPS if cls != 'whitenoise.runserver_nostatic'  # NOQA: F405
]

templating_app_index = INSTALLED_APPS.index(
    'mayan.apps.templating.apps.TemplatingApp'
)
INSTALLED_APPS.insert(
    templating_app_index + 1, 'mayan.apps.testing.apps.TestingApp'
)

LOGGING_LOG_FILE_PATH = '/tmp/mayan-errors.log'
LOGGING_LEVEL = 'WARNING'

# Remove middlewares not used for tests.
# Remove whitenoise from middlewares. Causes out of memory errors during test
# suit.
MIDDLEWARE = [
    cls for cls in MIDDLEWARE if cls not in [  # NOQA: F405
        'common.middleware.error_logging.ErrorLoggingMiddleware',
        'django.middleware.security.SecurityMiddleware',
        'corsheaders.middleware.CorsMiddleware',
        'django.middleware.csrf.CsrfViewMiddleware',
        'django.middleware.clickjacking.XFrameOptionsMiddleware',
        'common.middleware.timezone.TimezoneMiddleware',
        'common.middleware.ajax_redirect.AjaxRedirect',
        'whitenoise.middleware.WhiteNoiseMiddleware'
    ]
]

OCR_AUTO_OCR = False

# User a simpler password hasher.
PASSWORD_HASHERS = (
    'django.contrib.auth.hashers.MD5PasswordHasher',
)

SEARCH_BACKEND = 'mayan.apps.dynamic_search.tests.backends.TestSearchBackendProxy'

STORAGES['staticfiles'] = {
    'BACKEND': 'django.contrib.staticfiles.storage.StaticFilesStorage'
}

# Cache templates in memory.
TEMPLATES[0]['OPTIONS']['loaders'] = (  # NOQA: F405
    (
        'django.template.loaders.cached.Loader', (
            'django.template.loaders.filesystem.Loader',
            'django.template.loaders.app_directories.Loader',
        )
    ),
)

TESTING = True  # Silence the error logger for non critical `Http404` and `PermissionDenied`.
