"""Custom exceptions for the TCSPC toolkit."""


class TCSPCError(Exception):
    """Base exception for the TCSPC toolkit."""


class InvalidHistogramError(TCSPCError, ValueError):
    """Raised when a TCSPC histogram is physically or numerically invalid."""


class FeatureExtractionError(TCSPCError, ValueError):
    """Raised when requested histogram features are mathematically undefined."""
