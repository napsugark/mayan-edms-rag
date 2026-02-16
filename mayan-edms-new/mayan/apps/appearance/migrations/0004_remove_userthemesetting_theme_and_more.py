from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [
        ('appearance', '0003_auto_20210823_2114')
    ]

    operations = [
        migrations.RemoveField(model_name='userthemesetting', name='theme'),
        migrations.RemoveField(model_name='userthemesetting', name='user'),
        migrations.DeleteModel(name='Theme'),
        migrations.DeleteModel(name='UserThemeSetting')
    ]
