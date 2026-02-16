from mayan.apps.dependencies.classes import (
    JavaScriptDependency, PythonDependency
)

JavaScriptDependency(
    module=__name__, name='signature_pad', replace_list=[
        {
            'filename_pattern': '*.umd.js',
            'content_patterns': [
                {
                    'search': '//# sourceMappingURL=signature_pad.umd.js.map',
                    'replace': '',
                }
            ]
        }
    ], version_string='=4.2.0'
)
PythonDependency(
    module=__name__, name='CairoSVG', version_string='==2.8.2'
)
