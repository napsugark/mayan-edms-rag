def condition_user_is_not_super_user(context, resolved_object):
    return not condition_user_is_super_user(
        context=context, resolved_object=resolved_object
    )


def condition_user_is_super_user(context, resolved_object):
    is_staff = getattr(resolved_object, 'is_staff', False)
    is_super_user = getattr(resolved_object, 'is_superuser', False)
    return is_staff or is_super_user
