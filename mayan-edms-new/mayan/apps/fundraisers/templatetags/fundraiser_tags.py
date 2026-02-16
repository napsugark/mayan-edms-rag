from django.template import Library
from django.utils.safestring import mark_safe

from mayan.apps.views.http import URL

import nh3
import markdown
import requests

from ..literals import URL_FUNDRAISER_API

register = Library()


def markdown_render(source):
    md = markdown.Markdown(
        extensions=('attr_list', 'nl2br')
    )

    html = md.convert(source=source)

    html_clean = nh3.clean(
        attributes={
            'a': {'href'},
            'img': {'alt', 'class', 'src', 'style'}
        }, html=html
    )

    html_safe = mark_safe(s=html_clean)

    return html_safe


@register.simple_tag(name='fundraiser_message_fetch')
def tag_fundraiser_message_fetch(path):
    url = URL(url=URL_FUNDRAISER_API)
    url.path = path

    try:
        response = requests.get(url=url)
    except requests.exceptions.RequestException:
        return ''
    else:
        if response:
            response_json = response.json()

            body = response_json.get('body')
            title = response_json.get('title')

            body_safe = markdown_render(source=body)
            title_safe = markdown_render(source=title)

            return {
                'body': body_safe,
                'title': title_safe
            }
        else:
            return ''
