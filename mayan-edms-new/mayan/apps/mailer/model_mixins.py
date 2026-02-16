from furl import furl

from django.conf import settings
from django.core import mail
from django.utils.html import strip_tags
from django.utils.module_loading import import_string
from django.utils.translation import gettext_lazy as _

import mayan
from mayan.apps.templating.template_backends import Template

from .events import event_email_sent
from .literals import ERROR_LOG_DOMAIN_NAME
from .mailing_actions import ModelMailingAction
from .utils import split_recipient_list


class UserMailerBusinessLogicMixin:
    def get_connection(self):
        """
        Establishes a reusable connection to the server by loading the
        backend, initializing it, and the using the backend instance to get
        a connection.
        """
        backend_instance = self.get_backend_instance()
        connection_kwargs = backend_instance.get_connection_kwargs()

        return mail.get_connection(
            backend=backend_instance.class_path, **connection_kwargs
        )

    def send(
        self, to, _event_action_object=None, attachments=None, bcc=None,
        body='', cc=None, reply_to=None, subject='', user=None
    ):
        """
        Send a simple email. There is no document or template knowledge.
        attachments is a list of dictionaries with the keys:
        filename, content, and mimetype.
        """
        recipient_list = split_recipient_list(
            recipients=[to]
        )

        if cc:
            cc_list = split_recipient_list(
                recipients=[cc]
            )
        else:
            cc_list = None

        if bcc:
            bcc_list = split_recipient_list(
                recipients=[bcc]
            )
        else:
            bcc_list = None

        if reply_to:
            reply_to_list = split_recipient_list(
                recipients=[reply_to]
            )
        else:
            reply_to_list = None

        backend_data = self.get_backend_data()

        with self.get_connection() as connection:
            email_message = mail.EmailMultiAlternatives(
                bcc=bcc_list, body=strip_tags(body), cc=cc_list,
                connection=connection, from_email=backend_data.get('from'),
                reply_to=reply_to_list, subject=subject,
                to=recipient_list
            )

            for attachment in attachments or ():
                email_message.attach(
                    content=attachment['content'],
                    filename=attachment['filename'],
                    mimetype=attachment['mimetype']
                )

            email_message.attach_alternative(body, 'text/html')

        try:
            email_message.send()

        except Exception as exception:
            self.error_log.create(
                domain_name=ERROR_LOG_DOMAIN_NAME,
                text='{}; {}'.format(
                    exception.__class__.__name__, exception
                )
            )
        else:
            queryset_error_logs = self.error_log.filter(
                domain_name=ERROR_LOG_DOMAIN_NAME
            )
            queryset_error_logs.delete()

            event_email_sent.commit(
                action_object=_event_action_object, actor=user,
                target=self
            )

    def send_object(
        self, obj, to, as_attachment=False, bcc=None, body='', cc=None,
        object_name=None, organization_installation_url='', reply_to=None,
        subject='', user=None
    ):
        """
        Send an object file using this user mailing profile.
        """
        if as_attachment:
            action_name = 'attachment'
        else:
            action_name = 'link'

        model_mailing_action = ModelMailingAction.get_action_for_model(
            action_name=action_name, model=obj._meta.model
        )

        context_dictionary = {
            'link': furl(organization_installation_url).join(
                obj.get_absolute_url()
            ).tostr(),
            'object': obj,
            'object_name': object_name
        }

        body_template = Template(template_string=body)
        body_html_content = body_template.render(
            context=context_dictionary
        )

        subject_template = Template(template_string=subject)
        subject_text = strip_tags(
            subject_template.render(context=context_dictionary)
        )

        attachments = []
        if as_attachment:
            content_function = import_string(
                dotted_path=model_mailing_action.kwargs[
                    'content_function_dotted_path'
                ]
            )

            mime_type_function = import_string(
                dotted_path=model_mailing_action.kwargs[
                    'mime_type_function_dotted_path'
                ]
            )
            mime_type = mime_type_function(obj=obj)

            with content_function(obj=obj) as file_object:
                attachments.append(
                    {
                        'content': file_object.read(),
                        'filename': str(obj),
                        'mimetype': mime_type
                    }
                )

        return self.send(
            _event_action_object=obj, attachments=attachments, bcc=bcc,
            body=body_html_content, cc=cc, reply_to=reply_to,
            subject=subject_text, to=to, user=user
        )

    def test(self, to, user=None):
        """
        Send a test message to make sure the mailing profile settings are
        correct.
        """
        try:
            self.send(
                subject=_(message='Test email from %s') % mayan.__title__,
                to=to, user=user
            )
        except Exception as exception:
            self.error_log.create(
                domain_name=ERROR_LOG_DOMAIN_NAME,
                text='{}; {}'.format(
                    exception.__class__.__name__, exception
                )
            )
            if settings.DEBUG:
                raise
        else:
            queryset_error_logs = self.error_log.filter(
                domain_name=ERROR_LOG_DOMAIN_NAME
            )
            queryset_error_logs.delete()
