from django.shortcuts import resolve_url

from mozilla_django_oidc.views import OIDCAuthenticationCallbackView


class MayanOIDCAuthenticationCallbackView(OIDCAuthenticationCallbackView):
    @property
    def success_url(self):
        # Pull the next url from the session or settings--we don't need to
        # sanitize here because it should already have been sanitized.
        next_url = self.request.session.get('oidc_login_next', None)
        return next_url or resolve_url(
            self.get_settings('LOGIN_REDIRECT_URL', '/')
        )
