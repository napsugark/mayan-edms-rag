from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    dependencies = [
        ('contenttypes', '0002_remove_content_type_name'),
        ('messaging', '0002_remove_message_parent')
    ]

    operations = [
        migrations.AlterField(
            model_name='message',
            name='sender_content_type',
            field=models.ForeignKey(
                blank=True, null=True,
                on_delete=django.db.models.deletion.CASCADE,
                to='contenttypes.contenttype',
                verbose_name='Sender content type'
            )
        ),
        migrations.AlterField(
            model_name='message',
            name='sender_object_id',
            field=models.PositiveIntegerField(
                blank=True, null=True, verbose_name='Sender object ID'
            )
        )
    ]
