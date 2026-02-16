from django.utils.translation import gettext_lazy as _

from mayan.apps.app_manager.apps import MayanAppConfig
from mayan.apps.common.menus import menu_topbar

from .links import link_ajax_refresh


class AppearanceApp(MayanAppConfig):
    app_namespace = 'appearance'
    app_url = 'appearance'
    has_javascript_translations = True
    has_static_media = True
    has_tests = False
    name = 'mayan.apps.appearance'
    static_media_ignore_patterns = (
        'AUTHORS*', 'CHANGE*', 'CONTRIBUT*', 'CODE_OF_CONDUCT*', 'Grunt*',
        'MAINTAIN*', 'README*', '*.less', '*.md', '*.nupkg', '*.nuspec',
        '*.scss*', '*.sh', '*tests*', 'bower*', 'composer.json*',
        'demo*', 'grunt*', 'gulp*', 'install', 'less', 'package.json*',
        'package-lock*', 'test', 'tests', 'variable*', '*.xcf',
        'appearance/node_modules/@fancyapps/fancybox/docs/*',
        'appearance/node_modules/@fancyapps/fancybox/src/*',
        'appearance/node_modules/jquery/src/*',
        'appearance/node_modules/jquery-form/_config.yml',
        'appearance/node_modules/jquery-form/form.jquery.json',
        'appearance/node_modules/jquery-form/docs/*',
        'appearance/node_modules/jquery-form/src/*',
        'appearance/node_modules/select2/src/*',
        'appearance/node_modules/toastr/karma.conf.js',
        'appearance/node_modules/toastr/toastr.js',
        'appearance/node_modules/toastr/toastr-icon.png',
        'appearance/node_modules/toastr/nuget/*'
    )
    verbose_name = _(message='Appearance')

    def ready(self):
        super().ready()

        menu_topbar.bind_links(
            links=(link_ajax_refresh,)
        )
