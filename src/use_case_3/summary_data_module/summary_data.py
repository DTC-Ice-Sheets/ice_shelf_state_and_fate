"""Module to calculate simple summary data from time series."""

from pathlib import Path

import xarray as xr
import yaml

output_path = Path("/data/yak1/working/dtc_ice_sheets/dashboard_download_scratch")
initial_datasets_path = Path("/data/yak1/working/dtc_ice_sheets/initial_datasets")
config_dict_path = Path("src/use_case_3/dashboard/variable_mappings.yml")
config_dict_local_path = Path("src/use_case_3/dashboard/variable_mappings_local.yml")

# Load metadata about the available datasets and variables
with config_dict_path.open("r", encoding="utf-8") as f:
    config_dict = yaml.safe_load(f)
with config_dict_local_path.open("r", encoding="utf-8") as f:
    config_dict_local = config_dict.copy() | yaml.safe_load(f)


def summary_timeseries_zarr_to_json(input_path: Path) -> dict:
    """Get the summary data for a given ice shelf from a Zarr file."""
    summary_data = xr.open_dataset(input_path)
    metadata = config_dict_local["timeseries_summary"]

    plot_dict = {}
    for var in summary_data:
        if var in metadata:
            plot_dict[var] = {
                "timeseries_type": metadata[var]["timeseries_type"],
                "time": summary_data["time"].to_numpy().tolist(),
                "measurements": summary_data[var].to_numpy().tolist(),
                "uncertainties": summary_data[metadata[var]["measurement_uncertainties"]].to_numpy().tolist()
                if metadata[var]["measurement_uncertainties"] in summary_data
                else None,
                "uncertainty_type": metadata[var]["measurement_uncertainty_type"]
                if "measurement_uncertainties" in metadata[var]
                else None,
                "units": metadata[var]["units"],
            }

    return plot_dict
