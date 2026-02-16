from django.core import management

from .literals import COMMAND_NAME_MIGRATE


def handler_pre_initial_setup(sender, **kwargs):
    management.call_command(
        command_name=COMMAND_NAME_MIGRATE, interactive=False
    )


def handler_pre_upgrade(sender, **kwargs):
    management.call_command(
        command_name=COMMAND_NAME_MIGRATE, fake_initial=True,
        interactive=False
    )
