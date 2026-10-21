"""Module to calculate simple summary data from time series."""

from pathlib import Path

import xarray as xr
import yaml

config_dict_path = Path("src/use_case_3/timeseries_zarr_to_json_module/variable_mappings.yml")
config_dict_local_path = Path("src/use_case_3/timeseries_zarr_to_json_module/variable_mappings_local.yml")

# Load metadata about the available datasets and variables
with config_dict_path.open("r", encoding="utf-8") as f:
    config_dict = yaml.safe_load(f)
with config_dict_local_path.open("r", encoding="utf-8") as f:
    config_dict_local = config_dict.copy() | yaml.safe_load(f)


def timeseries_zarr_to_json(data: xr.Dataset) -> dict:
    """
    Convert time series data from xarray Dataset to JSON-compatible dictionary format.

    Parameters
    ----------
    data : xr.Dataset
        Input xarray Dataset containing time series variables with time coordinate
        and associated measurements.

    Returns
    -------
    dict
        Dictionary containing time series data for a set of variables with keys:
        - timeseries_type : str
            Type of time series data, i.e. whether it is cummulative or a derivative.
        - time : list
            List of time values
        - measurements : list
            List of measurement values
        - uncertainties : list or None
            List of uncertainty values if available, None otherwise
        - uncertainty_type : str or None
            Type of measurement uncertainty if available, None otherwise
        - units : str
            Units of measurement

    Notes
    -----
    Only processes variables that are present in the metadata configuration
    dictionary. Variables not in metadata are skipped.
    """
    metadata = config_dict_local["timeseries_summary"]

    plot_dict = {}
    for var in data:
        if var in metadata:
            plot_dict[var] = {
                "timeseries_type": metadata[var]["timeseries_type"],
                "time": data["time"].to_numpy().tolist(),
                "measurements": data[var].to_numpy().tolist(),
                "uncertainties": data[metadata[var]["measurement_uncertainties"]].to_numpy().tolist()
                if metadata[var]["measurement_uncertainties"] in data
                else None,
                "uncertainty_type": metadata[var]["measurement_uncertainty_type"]
                if "measurement_uncertainties" in metadata[var]
                else None,
                "units": metadata[var]["units"],
            }

    return plot_dict
