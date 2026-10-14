"""Domain errors mapped to OAuth-style HTTP error bodies by the API layer."""

from __future__ import annotations


class PlatformError(Exception):
    code = "invalid_request"
    status = 400

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class NotFoundError(PlatformError):
    code = "not_found"
    status = 404


class ConflictError(PlatformError):
    code = "conflict"
    status = 409


class UnauthorizedError(PlatformError):
    code = "unauthorized"
    status = 401


class ForbiddenError(PlatformError):
    code = "forbidden"
    status = 403


class ApprovalRequiredError(PlatformError):
    """Policy requires a human sign-off before this action may run."""

    code = "approval_required"
    status = 403

    def __init__(self, message: str, *, rules: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.rules = rules


class ContentRejectedError(PlatformError):
    """Input failed a content security control (injection, credentials)."""

    code = "content_rejected"
    status = 422


class DependencyUnavailableError(PlatformError):
    """A required dependency (identity fabric, model provider) cannot be reached."""

    code = "temporarily_unavailable"
    status = 503
