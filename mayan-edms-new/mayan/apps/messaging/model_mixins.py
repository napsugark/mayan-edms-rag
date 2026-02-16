import nh3

from django.utils.html import urlize
from django.utils.translation import gettext_lazy as _

from mayan.apps.templating.template_backends import Template


class MessageBusinessLogicMixin:
    def get_label(self):
        return Template(
            template_string='{{ instance.date_time }} @{{ instance.sender_object }} "{{ instance.subject }}"'
        ).render(
            context={'instance': self}
        )
    get_label.short_description = _(message='Label')

    def get_clean_body(self):
        output_linkified = urlize(
            autoescape=True, nofollow=False, text=self.body
        )

        output_cleaned = nh3.clean(
            html=output_linkified, link_rel='nofollow noopener noreferrer'
        )

        return output_cleaned

    def get_rendered_body(self):
        clean_body = self.get_clean_body()

        template = Template(template_string=clean_body)
        return template.render(
            context={'message': self}
        )

    def mark_read(self, user):
        self._event_actor = user
        self.read = True
        self.save(
            update_fields=('read',)
        )

    def mark_unread(self, user):
        self._event_actor = user
        self.read = False
        self.save(
            update_fields=('read',)
        )
