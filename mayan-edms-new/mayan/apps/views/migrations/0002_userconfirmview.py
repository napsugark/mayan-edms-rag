from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion

import mayan.apps.views.model_mixins


class Migration(migrations.Migration):
    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('views', '0001_initial')
    ]

    operations = [
        migrations.CreateModel(
            name='UserConfirmView',
            fields=[
                (
                    'id', models.AutoField(
                        auto_created=True, primary_key=True, serialize=False,
                        verbose_name='ID'
                    )
                ),
                (
                    'namespace', models.CharField(
                        db_index=True, max_length=200,
                        verbose_name='Namespace'
                    )
                ),
                (
                    'name', models.CharField(
                        db_index=True, help_text='Full name of the view '
                        'including the namespace.', max_length=200,
                        verbose_name='Name'
                    )
                ),
                (
                    'remember', models.BooleanField(
                        default=False, help_text='Remember the last '
                        'confirmation of the view.', verbose_name='Remember'
                    )
                ),
                (
                    'user', models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name='view_confirms',
                        to=settings.AUTH_USER_MODEL, verbose_name='User'
                    )
                )
            ],
            options={
                'verbose_name': 'User confirm view',
                'verbose_name_plural': 'User confirm view',
                'ordering': ('user__username', 'name'),
                'unique_together': {('user', 'name')}
            },
            bases=(
                mayan.apps.views.model_mixins.ModelMixinUserConfirmViewBusinessLogic,
                models.Model
            )
        )
    ]
