from django.db import migrations


def code_document_version_active_populate(apps, schema_editor):
    cursor_main = schema_editor.connection.create_cursor(
        name='document_version_active'
    )
    cursor_document_version_active_update = schema_editor.connection.cursor()

    query_main = '''
        SELECT
            {documents_document}.{id}, {documents_documentversion_active}.{id}
        FROM (
            SELECT
                {documents_documentversion_active_timestamp_max}.{document_id}, {documents_documentversion}.{id}
            FROM {documents_documentversion}, (
                SELECT
                    {document_id}, MAX(timestamp) as {timestamp_max}
                FROM
                    {documents_documentversion}
                WHERE
                    {active}
                GROUP BY
                    {document_id}, {active}
            ) AS {documents_documentversion_active_timestamp_max}
            WHERE {documents_documentversion}.{timestamp} = {documents_documentversion_active_timestamp_max}.{timestamp_max}
            AND {documents_documentversion}.{document_id} = {documents_documentversion_active_timestamp_max}.{document_id}
        ) as {documents_documentversion_active}
        INNER JOIN
            {documents_document}
        ON
            {documents_document}.{id} = {documents_documentversion_active}.{document_id}
    '''.format(
        active=schema_editor.connection.ops.quote_name(
            name='active'
        ),
        documents_document=schema_editor.connection.ops.quote_name(
            name='documents_document'
        ),
        document_id=schema_editor.connection.ops.quote_name(
            name='document_id'
        ),
        id=schema_editor.connection.ops.quote_name(
            name='id'
        ),
        documents_documentversion=schema_editor.connection.ops.quote_name(
            name='documents_documentversion'
        ),
        documents_documentversion_active=schema_editor.connection.ops.quote_name(
            name='documents_documentversion_active'
        ),
        documents_documentversion_active_timestamp_max=schema_editor.connection.ops.quote_name(
            name='documents_documentversion_active_timestamp_max'
        ),
        timestamp=schema_editor.connection.ops.quote_name(
            name='timestamp'
        ),
        timestamp_max=schema_editor.connection.ops.quote_name(
            name='timestamp_max'
        )
    )

    cursor_main.execute(query_main)

    FETCH_SIZE = 100000

    quoted_documents_document = schema_editor.connection.ops.quote_name(
        name='documents_document'
    )
    quoted_id = schema_editor.connection.ops.quote_name(name='id')
    quoted_version_active_id = schema_editor.connection.ops.quote_name(
        name='version_active_id'
    )

    while True:
        rows = cursor_main.fetchmany(FETCH_SIZE)
        for row in rows:
            query = f'''
                UPDATE {quoted_documents_document}
                    SET {quoted_version_active_id} = {row[1]}
                WHERE {quoted_id} = {row[0]};
            '''
            cursor_document_version_active_update.execute(query)

        if not rows:
            break


class Migration(migrations.Migration):
    dependencies = [
        ('documents', '0087_document_version_active')
    ]

    operations = [
        migrations.RunPython(
            code=code_document_version_active_populate,
            reverse_code=migrations.RunPython.noop
        )
    ]
