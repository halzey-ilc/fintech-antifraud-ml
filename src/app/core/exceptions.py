from __future__ import annotations


class AppError(Exception):
    """Base class for application-level errors."""


class ConfigError(AppError):
    """Raised when configuration is invalid or cannot be loaded."""


class DataValidationError(AppError):
    """Raised when input data fails validation checks."""
