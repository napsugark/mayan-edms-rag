from django.utils.translation import gettext_lazy as _

RELATIONSHIP_NO = 'no'
RELATIONSHIP_YES = 'yes'
RELATIONSHIP_CHOICES = (
    (RELATIONSHIP_NO, _(message='No')),
    (RELATIONSHIP_YES, _(message='Yes')),
)
