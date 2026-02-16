from django.template import Library

from ..models import Announcement

register = Library()


@register.inclusion_tag(
    filename='announcements/announcements.html', name='announcements'
)
def tag_announcements():
    return {'announcements': Announcement.objects.get_for_now()}
