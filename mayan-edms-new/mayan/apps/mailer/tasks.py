from django.apps import apps
from django.contrib.auth import get_user_model

from mayan.celery import app


@app.task(ignore_result=True)
def task_send_object(
    content_type_id, body, object_id, sender, subject, recipient,
    mailing_profile_id, as_attachment=False, object_name=None,
    organization_installation_url=None, user_id=None
):
    ContentType = apps.get_model(
        app_label='contenttypes', model_name='ContentType'
    )
    UserMailer = apps.get_model(
        app_label='mailer', model_name='UserMailer'
    )
    User = get_user_model()

    content_type = ContentType.objects.get(pk=content_type_id)
    obj = content_type.get_object_for_this_type(pk=object_id)

    mailing_profile = UserMailer.objects.get(pk=mailing_profile_id)

    if user_id:
        user = User.objects.get(pk=user_id)
    else:
        user = None

    mailing_profile.send_object(
        as_attachment=as_attachment, body=body, obj=obj,
        object_name=object_name,
        organization_installation_url=organization_installation_url,
        subject=subject, to=recipient, user=user
    )
