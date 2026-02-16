from docutils import nodes
from docutils.statemachine import ViewList
from sphinx.util.docutils import SphinxDirective
from sphinx.util.nodes import nested_parse_with_titles

from mayan.apps.smart_settings.classes import Setting
from mayan.apps.smart_settings.setting_clusters import SettingCluster
from mayan.apps.smart_settings.settings import setting_cluster


class DirectiveMayanSettingBase(SphinxDirective):
    _initialized = False

    @classmethod
    def initialize(cls):
        if not cls._initialized:
            SettingCluster.load_modules()
            cls._initialized = True

    def append_setting(self, docname, labels, setting, setting_node):
        # Paragraph with parsed content to display a backtick title and
        # label.
        node = nodes.paragraph()
        label_name = setting.global_name.replace('_', '-').lower()
        labels[
            'settings-{}'.format(label_name)
        ] = docname, label_name, setting.global_name
        nested_parse_with_titles(
            state=self.state, content=ViewList(
                initlist=[
                    '.. _{}:'.format(label_name),
                    '',
                    '``{}``'.format(setting.global_name),
                ]
            ), node=node
        )
        setting_node += node

        # Paragraph with a text to display the help text.
        node = nodes.paragraph()
        node += nodes.Text(data=setting.help_text, rawsource=setting.help_text)
        setting_node += node

        # Line "Default:"
        node = nodes.paragraph()
        node += nodes.Text(data='Default:', rawsource='Default:')
        setting_node += node

        # Paragraph with a literal block to display the default value.
        node = nodes.paragraph()
        default_value = '{}'.format(
            Setting.serialize_value(value=setting.default)
        )
        node += nodes.doctest_block(
            rawsource=default_value, text=default_value, childre=['python']
        )
        setting_node += node

        if setting.choices:
            # Line "Choices:"
            node = nodes.paragraph()
            node += nodes.Text(data='Choices:', rawsource='Choices:')
            setting_node += node

            # Paragraph with a literal block to display the default value.
            node = nodes.paragraph()
            value_choices = ','.join(setting.choices)
            node += nodes.doctest_block(
                rawsource=value_choices, text=value_choices,
                childre=['python']
            )
            setting_node += node

    def run(self):
        self.initialize()
        return self._run()


class DirectiveMayanSetting(DirectiveMayanSettingBase):
    final_argument_whitespace = True
    has_content = False
    option_spec = {}
    optional_arguments = 0
    required_arguments = 1

    def _run(self):
        labels = self.env.domaindata['std']['labels']
        docname = self.env.docname

        setting = setting_cluster.get_settings(
            global_name=self.arguments[0]
        )

        setting_node = nodes.bullet_list()

        self.append_setting(
            docname=docname, labels=labels, setting=setting,
            setting_node=setting_node
        )

        return [setting_node]


class DirectiveMayanSettingNamespace(DirectiveMayanSettingBase):
    final_argument_whitespace = True
    has_content = False
    option_spec = {}
    optional_arguments = 0
    required_arguments = 1

    def _run(self):
        labels = self.env.domaindata['std']['labels']
        docname = self.env.docname

        namespace = setting_cluster.get_namespace(
            name=self.arguments[0]
        )

        idb = nodes.make_id(
            string='settings-{}'.format(
                namespace.label.replace(' ', '-').replace('_', '-').lower()
            )
        )
        namespace_node = nodes.section(
            ids=[idb]
        )

        bullet_list_node = nodes.bullet_list()
        namespace_node += bullet_list_node

        for setting in namespace.get_setting_list():
            setting_node = nodes.list_item()
            bullet_list_node += setting_node

            self.append_setting(
                docname=docname, labels=labels, setting=setting,
                setting_node=setting_node
            )

        return [namespace_node]
