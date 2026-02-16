from mayan.apps.dependencies.classes import PythonDependency
from mayan.apps.dependencies.environments import environment_documentation

PythonDependency(
    environment=environment_documentation, module=__name__,
    name='Sphinx', version_string='==8.2.3'
)
PythonDependency(
    environment=environment_documentation, module=__name__,
    name='sphinx-sitemap', version_string='==2.8.0'
)
PythonDependency(
    environment=environment_documentation, module=__name__,
    name='sphinx_rtd_theme', version_string='==3.0.2'
)
PythonDependency(
    environment=environment_documentation, module=__name__,
    name='sphinxcontrib-spelling', version_string='==8.0.1'
)
