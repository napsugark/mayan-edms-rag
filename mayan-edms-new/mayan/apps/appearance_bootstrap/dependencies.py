from django.utils.translation import gettext_lazy as _

from mayan.apps.dependencies.classes import (
    GoogleFontDependency, JavaScriptDependency
)

GoogleFontDependency(
    label=_(message='Lato font'), module=__name__, name='lato',
    url='https://fonts.googleapis.com/css?family=Lato:400,700,400italic'
)
JavaScriptDependency(
    label=_(message='Bootstrap'), module=__name__, name='bootstrap',
    version_string='=3.4.1'
)
JavaScriptDependency(
    label=_(message='Bootswatch'), module=__name__, name='bootswatch',
    replace_list=[
        {
            'filename_pattern': 'bootstrap.*.css',
            'content_patterns': [
                {
                    'search': '"https://fonts.googleapis.com/css?family=Lato:400,700,400italic"',
                    'replace': '../../../google_fonts/lato/import.css',
                }
            ]
        }
    ], version_string='=3.4.1'
)
