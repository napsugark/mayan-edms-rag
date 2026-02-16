from .events import event_role_edited


def method_group_roles_add(self, queryset, user=None):
    for role in queryset:
        self.roles.add(role)
        event_role_edited.commit(
            action_object=self, actor=user, target=role
        )


def method_group_roles_remove(self, queryset, user=None):
    for role in queryset:
        self.roles.remove(role)
        event_role_edited.commit(
            action_object=self, actor=user, target=role
        )
