from mayan.apps.dependencies.environments import environment_build

from .classes import PythonDependency

PythonDependency(
    module=__name__, name='node-semver', version_string='==0.9.0'
)
PythonDependency(
    environment=environment_build, module=__name__, name='packaging',
    version_string='==25.0'
)
