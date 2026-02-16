from django.utils.encoding import force_bytes

TEST_EMAIL_ATTACHMENT_AND_INLINE = '''Subject: Test 03: inline and attachments
To: Renat Gilmanov
Content-Type: multipart/mixed; boundary=001a11c24d809f1525051712cc78

--001a11c24d809f1525051712cc78
Content-Type: multipart/related; boundary=001a11c24d809f1523051712cc77

--001a11c24d809f1523051712cc77
Content-Type: text/html; charset=UTF-8
Content-Transfer-Encoding: quoted-printable

<div dir=3D"ltr">Lorem ipsum dolor sit amet, consectetur adipiscing elit. P=
ellentesque odio urna, bibendum eu ultricies in, dignissim in magna. Vivamu=
s risus justo, viverra sed dapibus eu, laoreet eget erat. Sed pretium a urn=
a id pulvinar.<br><br><img src=3D"cid:ii_ia6yyemg0_14d9636d8ac7a587" height=
=3D"218" width=3D"320"><br>=E2=80=8B<br>Cras eu velit ac purus feugiat impe=
rdiet nec sit amet ipsum. Praesent gravida lobortis justo, nec tristique ve=
lit sagittis finibus. Suspendisse porta ante id diam varius, in cursus ante=
 luctus. Aenean a mollis mi. Pellentesque accumsan lacus sed erat vulputate=
, et semper tellus condimentum.<br><br>Best regards</div>

--001a11c24d809f1523051712cc77
Content-Type: image/png; name="test-01.png"
Content-Disposition: inline; filename="test-01.png"
Content-Transfer-Encoding: base64
Content-ID: <ii_ia6yyemg0_14d9636d8ac7a587>
X-Attachment-Id: ii_ia6yyemg0_14d9636d8ac7a587

iVBORw0KGgoAAAANSUhEUgAAAUAAAADaCAYAAADXGps7AAAABHNCSVQICAgIfAhkiAAAAAlwSFlz
AAALewAAC3sBSRnwgAAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAALnSURB
...
QCDLAIEsAwSyDBDIMkAgywCBLAMEsgwQyDJAIMsAgSwDBLIMEMgyQCDLAIEsAwSyDBDIMkAg6wK+
4gU280YtuwAAAABJRU5ErkJggg==
--001a11c24d809f1523051712cc77--
--001a11c24d809f1525051712cc78
Content-Type: image/png; name="test-02.png"
Content-Disposition: attachment; filename="test-02.png"
Content-Transfer-Encoding: base64
X-Attachment-Id: f_ia6yymei1'''
TEST_EMAIL_ATTACHMENT_AND_INLINE_EMAIL_TO = '"Renat Gilmanov"'
TEST_EMAIL_ATTACHMENT_AND_INLINE_SUBJECT = 'Test 03: inline and attachments'
TEST_EMAIL_ATTACHMENT_AND_INLINE_PART_0_CONTENT_TRANSFER_ENCODING = 'quoted-printable'
TEST_EMAIL_ATTACHMENT_AND_INLINE_PART_0_CONTENT_TYPE = 'text/html; charset="UTF-8"'
TEST_EMAIL_ATTACHMENT_AND_INLINE_PART_1_CONTENT_DISPOSITION = 'inline; filename="test-01.png"'
TEST_EMAIL_ATTACHMENT_AND_INLINE_PART_1_CONTENT_ID = '<ii_ia6yyemg0_14d9636d8ac7a587>'
TEST_EMAIL_ATTACHMENT_AND_INLINE_PART_1_CONTENT_TYPE = 'image/png; name="test-01.png"'
TEST_EMAIL_ATTACHMENT_AND_INLINE_PART_1_CONTENT_TRANSFER_ENCODING = 'base64'
TEST_EMAIL_ATTACHMENT_AND_INLINE_PART_1_X_ATTACHMENT_ID = 'ii_ia6yyemg0_14d9636d8ac7a587'

TEST_EMAIL_BASE64_FILENAME_ATTACHMENT_FILENAME = 'Ampelmännchen.txt'
TEST_EMAIL_BASE64_MESSAGE_ID = '<00000001.465619c9.1.00@BRN30055CCF4D76>'
TEST_EMAIL_BASE64_FILENAME = force_bytes(s='''From: noreply@example.com
To: test@example.com
Subject: Scan to E-mail Server Job
Date: Tue, 23 May 2017 23:03:37 +0200
Message-Id: {}
Mime-Version: 1.0
Content-Type: multipart/mixed;
    boundary="RS1tYWlsIENsaWVudA=="
X-Mailer: E-mail Client

This is multipart message.

--RS1tYWlsIENsaWVudA==
Content-Type: text/plain; charset=iso-8859-1
Content-Transfer-Encoding: quoted-printable

Sending device cannot receive e-mail replies.
--RS1tYWlsIENsaWVudA==
Content-Type: text/plain
Content-Transfer-Encoding: base64
Content-Disposition: attachment; filename="=?UTF-8?B?QW1wZWxtw6RubmNoZW4udHh0?="

SGFsbG8gQW1wZWxtw6RubmNoZW4hCg==

--RS1tYWlsIENsaWVudA==--'''.format(TEST_EMAIL_BASE64_MESSAGE_ID))
TEST_EMAIL_BASE64_FILENAME_FROM = 'noreply@example.com'
TEST_EMAIL_BASE64_FILENAME_SUBJECT = 'Scan to E-mail Server Job'

TEST_EMAIL_NO_CONTENT_TYPE_DATE = 'Mon, 09 Apr 2018 00:00:00 -0400'
TEST_EMAIL_NO_CONTENT_TYPE_DEVELIVERED_TO = 'test-sender@example.com'
TEST_EMAIL_NO_CONTENT_TYPE_FROM = 'Test Sender <test-sender@example.com>'
TEST_EMAIL_NO_CONTENT_TYPE_MESSAGE_ID = '<CAEAsyCbSF1Bk7CBuu6zp3Qs8=j2iUkNi3dPkGe6z40q4dmaogQ@mail.gmail.com>'
TEST_EMAIL_NO_CONTENT_TYPE_MIME_VERSION = '1.0'
TEST_EMAIL_NO_CONTENT_TYPE_RECEIVED = 'by 10.0.0.1 with HTTP; Mon, 9 Apr 2018 00:00:00 -0400 (AST)'
TEST_EMAIL_NO_CONTENT_TYPE_SUBJECT = 'Test message with no content type'
TEST_EMAIL_NO_CONTENT_TYPE_TO = 'test-receiver@example.com'
TEST_EMAIL_NO_CONTENT_TYPE_X_ORIGINATING_IP = '[10.0.0.1]'
TEST_EMAIL_NO_CONTENT_TYPE = '''MIME-Version: 1.0
Received: {email_received}
X-Originating-IP: [10.0.0.1]
Date: {email_date}
Delivered-To: {email_delivered_to}
Message-ID: {email_message_id}
Subject: {email_subject}
From: {email_from}
To: {email_to}

Test email without a content type'''.format(
    email_date=TEST_EMAIL_NO_CONTENT_TYPE_DATE,
    email_delivered_to=TEST_EMAIL_NO_CONTENT_TYPE_DEVELIVERED_TO,
    email_from=TEST_EMAIL_NO_CONTENT_TYPE_FROM,
    email_message_id=TEST_EMAIL_NO_CONTENT_TYPE_MESSAGE_ID,
    email_received=TEST_EMAIL_NO_CONTENT_TYPE_RECEIVED,
    email_subject=TEST_EMAIL_NO_CONTENT_TYPE_SUBJECT,
    email_to=TEST_EMAIL_NO_CONTENT_TYPE_TO
)
TEST_EMAIL_NO_CONTENT_TYPE_STRING = b'Test email without a content type'
TEST_EMAIL_INLINE_IMAGE = '''Subject: Test 01: inline only
To: Renat Gilmanov
Content-Type: multipart/related; boundary=089e0149bb0ea4e55c051712afb5

--089e0149bb0ea4e55c051712afb5
Content-Type: text/html; charset=UTF-8
Content-Transfer-Encoding: quoted-printable

<div dir=3D"ltr">Lorem ipsum dolor sit amet, consectetur adipiscing elit. P=
ellentesque odio urna, bibendum eu ultricies in, dignissim in magna. Vivamu=
s risus justo, viverra sed dapibus eu, laoreet eget erat. Sed pretium a urn=
a id pulvinar.<br><br><img src=3D"cid:ii_ia6yo3z92_14d962f8450cc6f1" height=
=3D"218" width=3D"320"><br>=E2=80=8B<br>Cras eu velit ac purus feugiat impe=
rdiet nec sit amet ipsum. Praesent gravida lobortis justo, nec tristique ve=
lit sagittis finibus. Suspendisse porta ante id diam varius, in cursus ante=
 luctus. Aenean a mollis mi. Pellentesque accumsan lacus sed erat vulputate=
, et semper tellus condimentum.<br><br>Best regards<br></div>

--089e0149bb0ea4e55c051712afb5
Content-Type: image/png; name="test-01.png"
Content-Disposition: inline; filename="test-01.png"
Content-Transfer-Encoding: base64
Content-ID: <ii_ia6yo3z92_14d962f8450cc6f1>
X-Attachment-Id: ii_ia6yo3z92_14d962f8450cc6f1

iVBORw0KGgoAAAANSUhEUgAAAUAAAADaCAYAAADXGps7AAAABHNCSVQICAgIfAhkiAAAAAlwSFlz
AAALewAAC3sBSRnwgAAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAALnSURB
...
QCDLAIEsAwSyDBDIMkAgywCBLAMEsgwQyDJAIMsAgSwDBLIMEMgyQCDLAIEsAwSyDBDIMkAg6wK+
4gU280YtuwAAAABJRU5ErkJggg==
--089e0149bb0ea4e55c051712afb5--'''
TEST_EMAIL_ZERO_LENGTH_ATTACHMENT = '''From rosarior@localhost Tue Apr  2 23:40:26 2019
Received: from rosarior (uid 1000)
    (envelope-from rosarior@localhost)
    id 2011e6
    by localhost (DragonFly Mail Agent v0.11);
    Tue, 02 Apr 2019 23:40:26 -0400
Message-ID: <22514.1554262827@localhost>
Mime-Version: 1.0
To: rosarior@localhost
Subject: mpack
Content-Type: multipart/mixed; boundary="-"
Date: Tue, 02 Apr 2019 23:40:26 -0400
From: <rosarior@localhost>

This is a MIME encoded message.  Decode it with "munpack"
or any other MIME reading software.  Mpack/munpack is available
via anonymous FTP in ftp.andrew.cmu.edu:pub/mpack/
---
Content-Type: application/octet-stream; name="zero"
Content-Transfer-Encoding: base64
Content-Disposition: inline; filename="zero"
Content-MD5: 1B2M2Y8AsgTpgAmY7PhCfg==


-----'''

TEST_EMAIL_SOURCE_PASSWORD = 'test_password'
TEST_EMAIL_SOURCE_USERNAME = 'test_username'

TEST_SOURCE_BACKEND_PATH_TEST_EMAIL = 'mayan.apps.source_emails.tests.source_backends.SourceBackendTestEmail'
TEST_SOURCE_BACKEND_PATH_TEST_EMAIL_IMAP = 'mayan.apps.source_emails.tests.source_backends.SourceBackendTestIMAPEmail'
TEST_SOURCE_BACKEND_PATH_TEST_EMAIL_POP3 = 'mayan.apps.source_emails.tests.source_backends.SourceBackendTestPOP3Email'
