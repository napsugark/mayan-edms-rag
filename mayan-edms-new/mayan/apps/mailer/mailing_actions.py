from mayan.apps.acls.classes import ModelPermission
from mayan.apps.events.classes import EventModelRegistry

from .exceptions import MailerError


class ModelMailingAction:
    _registry = {}
    arguments = ('manager_name', 'permission',)
    as_attachment = False
    name = None

    @classmethod
    def get_action_for_model(cls, model, action_name):
        try:
            model_actions = cls._registry[model]
        except KeyError:
            raise MailerError(
                'Model `{}` is not registered for emailing.'.format(model)
            )
        else:
            try:
                return model_actions[action_name]
            except KeyError:
                raise MailerError(
                    'Model `{}` is not registered for emailing action '
                    '`{}`.'.format(model, action_name)
                )

    def __init__(self, model, **kwargs):
        self.kwargs = {}

        arguments = self.get_arguments()

        for argument in arguments:
            try:
                value = kwargs.pop(argument)
            except KeyError:
                raise MailerError(
                    'Error registering mailer action `{}` for model `{}`. '
                    'Missing argument `{}`.'.format(
                        self.name, model, argument
                    )
                )

            self.kwargs[argument] = value

        if kwargs:
            raise MailerError(
                'Error registering mailer action `{}` for model `{}`. '
                'Too many keyword arguments `{}`.'.format(
                    self.name, model, kwargs
                )
            )

        self.__class__._registry.setdefault(
            model, {}
        )

        self.__class__._registry[model][self.name] = self

        EventModelRegistry.register(model=model)

        ModelPermission.register(
            model=model, permissions=(
                self.kwargs['permission'],
            )
        )

    def get_arguments(self):
        return self.arguments


class ModelMailingActionAttachment(ModelMailingAction):
    as_attachment = True
    name = 'attachment'

    def get_arguments(self):
        arguments = super().get_arguments()

        arguments += (
            (
                'content_function_dotted_path',
                'mime_type_function_dotted_path'
            )
        )

        return arguments


class ModelMailingActionLink(ModelMailingAction):
    name = 'link'
