from django.contrib.auth import get_user_model


def get_user_queryset(user=None):
    if user and (user.is_superuser or user.is_staff):
        queryset = get_user_model().objects.all()
    else:
        queryset = get_user_model().objects.filter(
            is_superuser=False, is_staff=False
        )

    return queryset.order_by('username')


def get_all_users_queryset(user=None):
    queryset = get_user_model().objects.all()

    return queryset.order_by('username')
