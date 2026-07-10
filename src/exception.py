class CustomException(Exception):
    """Custom exception for project-level failures."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message
