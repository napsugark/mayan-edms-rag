from mayan.apps.sources.source_backend_actions.interfaces import (
    SourceBackendActionInterface, SourceBackendActionInterfaceTask
)

from .arguments import argument_dry_run


class SourceBackendActionMixinPeriodic:
    class Interface:
        class Model(SourceBackendActionInterface):
            class Argument:
                dry_run = argument_dry_run

            def process_interface_context(self):
                super().process_interface_context()

                self.action_kwargs['dry_run'] = self.context['dry_run']

        class Task(SourceBackendActionInterfaceTask):
            class Argument:
                dry_run = argument_dry_run

            def process_interface_context(self):
                super().process_interface_context()

                self.action_kwargs['dry_run'] = self.context['dry_run']
