from django.conf import settings
from django.utils.translation import gettext_lazy as _

from mayan.apps.dynamic_search.search_models import SearchModel

from .permissions import permission_group_view, permission_user_view
from .querysets import get_user_queryset

search_model_group = SearchModel(
    app_label='auth', label=_(message='Group'), model_name='Group',
    permission=permission_group_view,
    serializer_path='mayan.apps.user_management.serializers.GroupSerializer'
)

search_model_group.add_model_field(
    field='name', label=_(message='Name')
)

user_app, user_model = settings.AUTH_USER_MODEL.split('.')

search_model_user = SearchModel(
    app_label=user_app, label=_(message='User'), model_name=user_model,
    permission=permission_user_view, queryset=get_user_queryset,
    serializer_path='mayan.apps.user_management.serializers.UserSerializer'
)

search_model_user.add_model_field(
    field='first_name', label=_(message='First name')
)
search_model_user.add_model_field(
    field='email', label=_(message='Email')
)
search_model_user.add_model_field(
    field='groups__name', label=_(message='Groups')
)
search_model_user.add_model_field(
    field='last_name', label=_(message='Last name')
)
search_model_user.add_model_field(
    field='username', label=_(message='Username')
)
