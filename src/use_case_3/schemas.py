"""Here we define the schemas for package entrypoints and outputs.

This helps us ensure the intended usage of the package is well defined and well understood.
"""

# schemas/entry_point_schemas.py
from pathlib import Path

from pydantic import BaseModel, Field

from use_case_3.timeseries_module.timeseries_enums import (
    AggregationPeriod,
    AggregationStrategy,
    NormalisationType,
    TimeseriesRegion,
    TimeseriesType,
    UncertaintyType,
)


# Example schema for the example entry point
class ExampleEntryPointSchema(BaseModel):
    """Example schema."""

    integer_a: int = Field(..., description="The first integer.")
    integer_b: int = Field(..., description="The second integer.")


class TimeseriesModuleEntryPointSchema(BaseModel):
    """Schema for the timeseries module entry point."""

    label: str = Field(..., description="Label for the timeseries.")
    dataset_path: Path = Field(..., description="Path to the input dataset .zarr file.")
    output_path: Path = Field(..., description="Path to save the output .zarr file.")
    measurement_name: str = Field(..., description="Name of the measurement to extract from the dataset.")
    measurement_uncertainty_name: str | None = Field(
        None, description="Name of the uncertainty to extract from the dataset."
    )
    masks_json_path: Path | None = Field(None, description="Path to the masks JSON file.")
    apply_mask: bool = Field(
        True,
        description=(
            "Whether the mask will be used to filter the data. Otherwise, the mask is only used to calculate coverage."
        ),
    )
    timeseries_type: TimeseriesType = Field(
        TimeseriesType.CUMULATIVE,
        description="Type of timeseries to compute (cumulative or derivative).",
    )
    timeseries_region: TimeseriesRegion = Field(
        TimeseriesRegion.ICE_SHELF,
        description="Region type for the timeseries (iceshelf, ocean, or climate).",
    )
    aggregate: AggregationStrategy = Field(
        AggregationStrategy.MEAN,
        description="Aggregation strategy to apply when multiple measurements fall within the same time bin.",
    )
    period: AggregationPeriod | None = Field(
        AggregationPeriod.MONTH,
        description=(
            "Time period for aggregation. If None, a single value will be computed for the entire timeseries."
        ),
    )
    uncertainty: UncertaintyType | None = Field(None, description="The statistics type describing the uncertainty.")
    independent: bool = Field(False, description="Whether the uncertainties are independent or not.")
    t0_normalisation: NormalisationType | None = Field(None, description="The type of normalisation to apply, if any.")


class TimeseriesPreprocessorModuleEntryPointSchema(BaseModel):
    """Schema for the timeseries preprocessor module entry point."""

    input_paths: list[Path] = Field(
        ...,
        description="List of paths to the input timeseries .zarr files",
    )
    output_path: Path = Field(..., description="Path to save the output .zarr file.")
    time_period: AggregationPeriod = Field(
        AggregationPeriod.MONTH,
        description=(
            "Time period for aggregation. If None, a single value will be computed for the entire timeseries."
        ),
    )
    aggregation: AggregationStrategy = Field(
        AggregationStrategy.MEAN,
        description="Aggregation strategy to apply when multiple measurements fall within the same time bin.",
    )


class CovariateAnalyserModuleEntryPointSchema(BaseModel):
    """Schema for the covariate analyser module entry point."""

    input_path: Path = Field(..., description="Path to the input .zarr file containing pre-processed timeseries.")
    output_path: Path = Field(
        ..., description="Path to save the output plot image file. Can use any filetype supported by matplotlib."
    )
