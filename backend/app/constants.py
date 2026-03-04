"""Application-wide constants."""

# Tagging constants
STABILITY_THRESHOLD = 0.75  # Minimum stability score before flagging for review
DEFAULT_N_SAMPLES = 4  # Number of LLM samples per response for stability scoring
DEFAULT_TAG_THRESHOLD = 2  # Minimum votes needed for a tag to be included

# Pagination constants
MAX_PAGE_SIZE = 200  # Maximum items per page for paginated endpoints
DEFAULT_PAGE_SIZE = 50  # Default items per page

# Job queue constants
DEFAULT_POLLING_INTERVAL = 60  # Seconds between batch status polls
BATCH_JOB_TIMEOUT = 86400  # 24 hours in seconds for batch jobs
