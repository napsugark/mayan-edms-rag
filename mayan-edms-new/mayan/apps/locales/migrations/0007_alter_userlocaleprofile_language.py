from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ('locales', '0006_alter_userlocaleprofile_timezone')
    ]

    operations = [
        migrations.AlterField(
            model_name='userlocaleprofile', name='language',
            field=models.CharField(
                choices=[
                    ('sq', 'Albanian'),
                    ('ar', 'Arabic'),
                    ('ar-eg', 'Arabic (Egypt)'),
                    ('bs', 'Bosnian'),
                    ('bg', 'Bulgarian'),
                    ('ca', 'Catalan'),
                    ('zh-cn', 'Chinese (China)'),
                    ('zh-hans', 'Chinese (Simplified)'),
                    ('zh-tw', 'Chinese (Taiwan)'),
                    ('hr', 'Croatian'),
                    ('cs', 'Czech'),
                    ('da', 'Danish'),
                    ('nl', 'Dutch'),
                    ('en', 'English'),
                    ('fr', 'French'),
                    ('de-at', 'German (Austria)'),
                    ('de-de', 'German (Germany)'),
                    ('el', 'Greek'),
                    ('he-il', 'Hebrew (Israel)'),
                    ('hu', 'Hungarian'),
                    ('id', 'Indonesian'),
                    ('it', 'Italian'),
                    ('lv', 'Latvian'),
                    ('mn-mn', 'Mongolian (Mongolia)'),
                    ('fa', 'Persian'),
                    ('pl', 'Polish'),
                    ('pt', 'Portuguese'),
                    ('pt-br', 'Portuguese (Brazil)'),
                    ('ro-ro', 'Romanian (Romania)'),
                    ('ru', 'Russian'),
                    ('sl', 'Slovenian'),
                    ('es', 'Spanish'),
                    ('es-mx', 'Spanish (Mexico)'),
                    ('es-pr', 'Spanish (Puerto Rico)'),
                    ('th', 'Thai'),
                    ('tr', 'Turkish'),
                    ('tr-tr', 'Turkish (Turkey)'),
                    ('uk', 'Ukrainian'),
                    ('vi', 'Vietnamese')
                ], max_length=8, verbose_name='Language'
            )
        )
    ]
