"""Public exceptions for Veldra API."""


class VeldraError(Exception):
    """Base class for all user-facing Veldra errors."""


class VeldraValidationError(VeldraError):
    """Raised when user-provided config or input is invalid."""


class VeldraArtifactError(VeldraError):
    """Raised when artifact save/load operations fail."""


class VeldraNotImplementedError(VeldraError):
    """Raised by API surfaces that are intentionally not implemented yet."""
