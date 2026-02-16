import os

# RAG Integration Settings

# Base URL for the external RAG API service.
# Should be the Docker container hostname when running in Docker.
RAG_API_BASE_URL = os.environ.get(
    'MAYAN_RAG_API_BASE_URL', 'http://rag:8000'
)

# Timeout in seconds for RAG API requests.
# Increase for large documents that need OCR/embedding processing.
RAG_API_TIMEOUT = int(
    os.environ.get('MAYAN_RAG_API_TIMEOUT', '300')
)

# Enable automatic indexing when a new document version is created.
# When True, new uploads are automatically sent to RAG after OCR completes.
RAG_AUTO_INDEX_ENABLED = os.environ.get(
    'MAYAN_RAG_AUTO_INDEX_ENABLED', 'true'
).lower() in ('true', '1', 'yes')

# Delay in seconds before sending a new document version to RAG.
# This gives OCR time to finish. Adjust based on your average OCR duration.
RAG_AUTO_INDEX_DELAY_SECONDS = int(
    os.environ.get('MAYAN_RAG_AUTO_INDEX_DELAY_SECONDS', '120')
)

# Maximum number of retries if OCR is not yet complete when the
# auto-index task runs.
RAG_AUTO_INDEX_MAX_RETRIES = int(
    os.environ.get('MAYAN_RAG_AUTO_INDEX_MAX_RETRIES', '5')
)

# Retry delay in seconds when OCR is not yet complete.
RAG_AUTO_INDEX_RETRY_DELAY = int(
    os.environ.get('MAYAN_RAG_AUTO_INDEX_RETRY_DELAY', '60')
)
