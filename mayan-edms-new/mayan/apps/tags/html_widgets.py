from django.template.loader import render_to_string


def widget_single_tag(tag):
    return render_to_string(
        template_name='tags/tag_widget.html', context={'tag': tag}
    )
