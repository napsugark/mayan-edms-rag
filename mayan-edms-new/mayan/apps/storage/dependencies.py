from mayan.apps.dependencies.classes import PythonDependency
from mayan.apps.dependencies.environments import environment_testing

from mayan.settings.literals import PYTHON_PSUTIL_VERSION

PythonDependency(
    module=__name__, name='boto3', version_string='==1.40.24'
)
PythonDependency(
    module=__name__, name='django-storages', version_string='==1.14.6'
)
PythonDependency(
    module=__name__, name='extract-msg', version_string='==0.55.0'
)
PythonDependency(
    module=__name__, name='google-cloud-storage', version_string='==3.3.1'
)
PythonDependency(
    environment=environment_testing, module=__name__, name='psutil',
    version_string='=={}'.format(
        PYTHON_PSUTIL_VERSION
    )
)
PythonDependency(
    module=__name__, name='pycryptodome', version_string='==3.23.0'
)
