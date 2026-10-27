Quick Start
===========

The package provides five command-line entrypoints. From a cloned repository with
the project dependencies installed (see the installation instructions in the
project README), run each entrypoint with Poetry:

.. code-block:: console

   poetry run python -m use_case_3.timeseries_module_entrypoint --help

Replace ``--help`` with the arguments shown below to run a command. If the
package is installed in your active Python environment, you can omit
``poetry run``. Each entrypoint also accepts ``--help`` to list its arguments
and defaults. Input Zarr stores may be local paths or ``http://``, ``https://``,
or ``s3://`` URLs. Output paths are local directories.

The commands can be combined into a processing workflow:

1. Extract a timeseries from one or more source datasets.
2. Optionally preprocess multiple timeseries together.
3. Analyse covariates in the preprocessed store.

Dataset summaries and timeseries-to-JSON conversion are separate operations and
can be run on their own.

Timeseries extraction
---------------------

Run ``timeseries_module_entrypoint`` to extract a measurement from each input
dataset and save a timeseries Zarr store. Repeat ``--dataset_paths`` for each
input dataset:

.. code-block:: console

   poetry run python -m use_case_3.timeseries_module_entrypoint \
      --label "Ice velocity" \
      --dataset_paths data/velocity_2020.zarr \
      --dataset_paths data/velocity_2021.zarr \
      --output_path output/velocity.zarr \
      --measurement_name velocity

``--measurement_name`` names the measurement variable in the input datasets.
Optionally provide ``--measurement_uncertainty_name`` if the input also
contains an uncertainty variable, and ``--mask_json_path`` to supply a GeoJSON
mask. The default timeseries type is ``cumulative`` (use ``derivative`` for a
derivative); aggregation defaults to ``mean`` by ``month``. The available
aggregation strategies are ``mean``, ``median``, ``max``, ``min``, ``std``,
``IQR``, and ``None``; aggregation periods are ``year``, ``quarter``, ``month``,
and ``day``.

Preprocessing timeseries
------------------------

Run ``timeseries_preprocessor_module_entrypoint`` to combine one or more
timeseries Zarr stores into the format expected by covariate analysis. Repeat
``--input_paths`` for each store:

.. code-block:: console

   poetry run python -m use_case_3.timeseries_preprocessor_module_entrypoint \
      --input_paths output/velocity.zarr \
      --input_paths output/temperature.zarr \
      --output_path output/preprocessed_timeseries.zarr \
      --time_period year \
      --aggregation mean

If ``--time_period`` is omitted, the lowest frequency in the input timeseries
is used. The aggregation defaults to ``mean`` and accepts ``mean``, ``median``,
``max``, ``min``, and ``std``.

Covariate analysis
------------------

Run ``covariate_analyser_module_entrypoint`` on the preprocessed store to write
analysis results as JSON:

.. code-block:: console

   poetry run python -m use_case_3.covariate_analyser_module_entrypoint \
      --input_path output/preprocessed_timeseries.zarr \
      --output_path output/covariate_analysis.json \
      --analysis_type corr

The analysis type defaults to ``corr``. Other supported values are
``cross_corr`` and ``causal``.

Dataset summary
---------------

Run ``dataset_summary_module_entrypoint`` to create a JSON summary of a GeoZarr
dataset. ``--max_res`` controls the maximum summary resolution and defaults to
``20``:

.. code-block:: console

   poetry run python -m use_case_3.dataset_summary_module_entrypoint \
      --input_path data/dataset.zarr \
      --output_path output/dataset_summary.json \
      --max_res 20

Convert a timeseries Zarr store to JSON
---------------------------------------

Run ``timeseries_zarr_to_json_module_entrypoint`` to convert a timeseries Zarr
store into its JSON representation:

.. code-block:: console

   poetry run python -m use_case_3.timeseries_zarr_to_json_module_entrypoint \
      --input_path output/velocity.zarr \
      --output_path output/velocity.json