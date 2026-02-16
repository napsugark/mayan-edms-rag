import hashlib
import json
import logging

import requests

from django.conf import settings
from django.utils import timezone

from mayan.apps.acls.models import AccessControlList
from mayan.apps.documents.permissions import permission_document_view

from .models import RAGDocumentVersionSync

logger = logging.getLogger(name=__name__)


def compute_document_file_hash(document_file):
    """
    Compute a SHA-256 hash of the document file's binary contents.

    This hash is stored alongside the sync record so that re-syncs
    are triggered when the underlying file content changes (e.g. a new
    file is uploaded to the same document version).

    Reads the file in 64 KB chunks to avoid loading multi-GB files
    into memory.
    """
    sha256 = hashlib.sha256()
    file_object = document_file.open()
    try:
        while True:
            chunk = file_object.read(65536)
            if not chunk:
                break
            sha256.update(chunk)
    finally:
        file_object.close()

    return sha256.hexdigest()


def check_rag_collection_status():
    """
    Query the RAG service health / status endpoint to determine
    whether the vector collection exists and how many documents
    it contains.

    Returns a dict with:
        'available': bool — whether the RAG service responded
        'document_count': int or None — docs in collection
        'error': str or None — error message if unavailable

    Degrades gracefully: if the RAG service doesn't expose a
    /health endpoint, returns available=True with document_count=None
    (we simply cannot verify).
    """
    base_url = getattr(settings, 'RAG_API_BASE_URL', 'http://rag:8000')

    for endpoint in ('/health', '/'):
        try:
            response = requests.get(
                url='{}{}'.format(base_url, endpoint), timeout=10
            )
            if response.status_code == 200:
                try:
                    data = response.json()
                    return {
                        'available': True,
                        'document_count': data.get(
                            'document_count', data.get('indexed_documents')
                        ),
                        'error': None
                    }
                except (ValueError, KeyError):
                    return {
                        'available': True,
                        'document_count': None,
                        'error': None
                    }
        except requests.exceptions.RequestException:
            continue

    # Could not reach any endpoint — service may be down.
    return {
        'available': False,
        'document_count': None,
        'error': 'RAG service is unreachable.'
    }


def document_version_needs_sync(document_version):
    """
    Determine whether a document version needs to be synced to RAG.

    Checks the sync record's boolean flag AND content hash to detect:
    1. Never-synced documents
    2. Documents whose file content changed since last sync
    3. Legacy records that pre-date the hash field

    Returns (needs_sync: bool, current_hash: str).
    """
    document_file = document_version.document.file_latest
    if not document_file:
        return False, None

    try:
        current_hash = compute_document_file_hash(
            document_file=document_file
        )
    except Exception:
        logger.exception(
            'Error computing hash for document version %s.',
            document_version.pk
        )
        # If we can't compute the hash, default to needing sync
        # (fail open — better to re-send than to silently skip).
        return True, None

    try:
        sync_record = RAGDocumentVersionSync.objects.get(
            document_version=document_version
        )
        return sync_record.needs_sync(current_hash=current_hash), current_hash
    except RAGDocumentVersionSync.DoesNotExist:
        return True, current_hash


def get_allowed_user_ids_for_document(document):
    """
    Return a list of user IDs that have view permission on the given
    document. Uses the ACL system to determine access.
    """
    from django.contrib.auth import get_user_model

    User = get_user_model()

    all_users = User.objects.filter(is_active=True)

    allowed_user_ids = []
    for user in all_users:
        filtered = AccessControlList.objects.restrict_queryset(
            permission=permission_document_view,
            queryset=document.__class__.valid.filter(pk=document.pk),
            user=user
        )
        if filtered.exists():
            allowed_user_ids.append(user.pk)

    return allowed_user_ids


def send_document_version_to_rag(document_version, content_hash=None):
    """
    Send a document version to the external RAG service for indexing.

    Retrieves the binary file from the latest document file, determines
    which users have view permission, and POSTs a multipart/form-data
    request to the RAG /index endpoint.

    If content_hash is provided, it is stored in the sync record.
    Otherwise, the hash is computed from the file (costs one extra read).

    Returns True on success, False on failure.
    """
    document = document_version.document

    # Get the file content from the document's latest file
    document_file = document.file_latest
    if not document_file:
        logger.warning(
            'Document version %s has no associated file, skipping.',
            document_version.pk
        )
        return False

    try:
        file_object = document_file.open()
    except Exception:
        logger.exception(
            'Error opening file for document version %s.',
            document_version.pk
        )
        return False

    try:
        # Determine allowed users
        allowed_user_ids = get_allowed_user_ids_for_document(
            document=document
        )

        # Build the multipart/form-data payload
        url = '{}/index'.format(
            getattr(settings, 'RAG_API_BASE_URL', 'http://rag:8000')
        )
        timeout = getattr(settings, 'RAG_API_TIMEOUT', 30)

        files = {
            'file': (
                document_file.filename or 'document',
                file_object,
                document_file.mimetype or 'application/octet-stream'
            )
        }
        data = {
            'document_id': document.pk,
            'document_version_id': document_version.pk,
            'allowed_users': json.dumps(allowed_user_ids)
        }

        response = requests.post(
            url=url, data=data, files=files, timeout=timeout
        )
        response.raise_for_status()

        # Compute content hash if not already provided.
        if not content_hash:
            try:
                content_hash = compute_document_file_hash(
                    document_file=document_file
                )
            except Exception:
                logger.warning(
                    'Could not compute content hash for document '
                    'version %s. Sync record will lack hash.',
                    document_version.pk
                )

        # Mark as synced with content hash for change detection.
        sync_record, created = RAGDocumentVersionSync.objects.get_or_create(
            document_version=document_version
        )
        sync_record.rag_synced = True
        sync_record.content_hash = content_hash
        sync_record.last_synced = timezone.now()
        sync_record.save(
            update_fields=('rag_synced', 'content_hash', 'last_synced')
        )

        logger.info(
            'Successfully synced document version %s to RAG.',
            document_version.pk
        )
        return True

    except requests.exceptions.RequestException:
        logger.exception(
            'Error sending document version %s to RAG service.',
            document_version.pk
        )
        return False
    except Exception:
        logger.exception(
            'Unexpected error syncing document version %s.',
            document_version.pk
        )
        return False
    finally:
        file_object.close()


def ai_assisted_search(user, query):
    """
    Send a synchronous query to the RAG /query endpoint and return
    the response containing an answer and matching document IDs.

    Returns a dict with keys 'answer' and 'results', or None on error.
    """
    url = '{}/query'.format(
        getattr(settings, 'RAG_API_BASE_URL', 'http://rag:8000')
    )
    timeout = getattr(settings, 'RAG_API_TIMEOUT', 30)

    payload = {
        'user_id': user.pk,
        'query': query
    }

    try:
        response = requests.post(
            url=url, json=payload, timeout=timeout
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
        logger.exception(
            'Error querying RAG service for user %s.', user.pk
        )
        return None
    except (ValueError, KeyError):
        logger.exception(
            'Invalid response from RAG service for user %s.', user.pk
        )
        return None
