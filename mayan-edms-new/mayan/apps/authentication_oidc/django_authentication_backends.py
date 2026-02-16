from mozilla_django_oidc.auth import OIDCAuthenticationBackend


class DjangoAuthenticationBackendOIDC(OIDCAuthenticationBackend):
    def create_user(self, claims):
        user = super().create_user(claims=claims)

        user._event_ignore = True

        user.first_name = claims.get('given_name', '')
        user.last_name = claims.get('family_name', '')
        user.save(
            update_fields=('first_name', 'last_name')
        )

        return user

    def update_user(self, user, claims):
        user = super().update_user(user=user, claims=claims)

        user._event_ignore = True

        user.first_name = claims.get('given_name', '')
        user.last_name = claims.get('family_name', '')
        user.save(
            update_fields=('first_name', 'last_name')
        )

        return user
