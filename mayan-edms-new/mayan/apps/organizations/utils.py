from mayan.apps.views.http import URL

from .settings import (
    setting_organization_installation_url, setting_organization_url_base_path
)


def get_organization_installation_url(request=None):
    installation_url = setting_organization_installation_url.value
    installation_path = setting_organization_url_base_path.value

    if installation_url:
        return URL(
            path=installation_path, url=installation_url
        ).to_string()
    elif request:
        return URL(
            netloc=request.get_host(), path=installation_path,
            port=request.get_port(), scheme=request.scheme
        ).to_string()
    else:
        return ''
