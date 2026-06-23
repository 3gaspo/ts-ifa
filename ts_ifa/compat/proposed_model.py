"""Compatibility imports for the renamed TS-IFA model."""

from ..models.ts_ifa import TSIFAConfig, TimeSeriesInformedForecastingAdapter

ProposedModelConfig = TSIFAConfig

__all__ = [
    "TSIFAConfig",
    "TimeSeriesInformedForecastingAdapter",
    "ProposedModelConfig",
]
