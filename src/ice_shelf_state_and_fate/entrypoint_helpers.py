"""
Helper functions for entrypoint scripts.

Helps to optimise loading time by avoiding imports of large dependencies from helper_functions.py.
"""

from pathlib import Path
from typing import Annotated

from cyclopts import Parameter


def path_or_web_url_validator(type_: object, path_or_url: Path | str) -> str:  # noqa: ARG001
    """Validate that the provided path is either a valid local path or a web URL.

    NOTE: Path("") resolves to the current directory (".") - which is considered a valid path.

    Parameters
    ----------
    type_ : object
        The type of the argument being validated - unused, purely required for cyclopts validator compatibility.
    path_or_url : Path | str
        The candidate path or URL to validate.

    Returns
    -------
    str
        The validated path or URL as a string.

    Raises
    ------
    ValueError
        If the provided path is neither a valid local path nor a web URL like string.
    """
    if not Path(path_or_url).exists() and not str(path_or_url).startswith(("http://", "https://", "s3://")):
        raise ValueError(f"The provided path '{path_or_url}' is neither a valid local path nor a web URL.")
    return str(path_or_url)


# Alias for use in cyclopts Parameter definitions.
# Override the sphinx documentation entry in the lines below since cyclopts doesn't produce nice results for
# custom validators:
#: A local path or web URL, represented as a string.
#:
#: Validated by :func:`path_or_web_url_validator`.
#:
#: :meta hide-value:
type ZarrType = Annotated[str, Parameter(validator=path_or_web_url_validator)]
