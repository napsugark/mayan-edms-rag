from mayan.apps.authentication.forms import AuthenticationFormBase


class AuthenticationFormOIDC(AuthenticationFormBase):
    def __init__(self, **kwargs):
        kwargs.update(
            {
                'data': None, 'files': None
            }
        )
        super().__init__(**kwargs)
