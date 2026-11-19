"""Separate training methods, embedded dispersion and runtime corrections."""

from __future__ import annotations

from typing import Any

DISPERSION_METHODS = frozenset({"none", "D3", "D4", "VV10", "MBD"})


def dispersion_method(value: Any) -> str | None:
    """
    Validate and normalize a dispersion declaration.

    Parameters
    ----------
    value
        Scheme name, or None for unknown treatment.

    Returns
    -------
    str | None
        Canonical scheme name. Missing metadata never means no dispersion.
    """
    if value is None:
        return None
    method = value
    names = {name.casefold(): name for name in DISPERSION_METHODS}
    if not isinstance(method, str) or method.casefold() not in names:
        raise ValueError(f"Unknown dispersion method: {method!r}")
    return names[method.casefold()]


def method_metadata(
    level: str | None,
    dispersion: Any = None,
    *,
    trained_on_dispersion: bool | None = None,
) -> dict[str, Any]:
    """
    Read separate method fields, accepting legacy explicit dispersion suffixes.

    Parameters
    ----------
    level
        Declared training or reference method.
    dispersion
        Explicit dispersion_method declaration, if available.
    trained_on_dispersion
        Model training flag. False defaults an unspecified scheme to none.
        Omit for benchmark reference metadata.

    Returns
    -------
    dict[str, Any]
        Base level and dispersion scheme. Intrinsic functional names stay intact.
    """
    scheme = dispersion_method(dispersion)
    for suffix in ("D3", "D4", "VV10", "MBD"):
        if level and level.endswith(f"+{suffix}"):
            if scheme is not None and scheme != suffix:
                raise ValueError("Conflicting level_of_theory and dispersion fields")
            level = level[: -(len(suffix) + 1)]
            scheme = suffix
            break
    if scheme is None and trained_on_dispersion is False:
        scheme = "none"
    return {"level_of_theory": level, "dispersion": scheme}


class DispersionNameMap(dict[str, str]):
    """Display names identifying models corrected by the shared D3 helper."""


def effective_method(config: dict[str, Any]) -> dict[str, Any]:
    """
    Derive the comparison method from model and benchmark configuration.

    Parameters
    ----------
    config
        Model metadata, including runtime_dispersion_method set by analysis.

    Returns
    -------
    dict[str, Any]
        Base method and dispersion, without requiring calculation records.
    """
    method = method_metadata(
        config.get("level_of_theory"),
        config.get("dispersion_method"),
        trained_on_dispersion=config.get("trained_on_dispersion"),
    )
    if (
        config.get("runtime_dispersion_method") == "D3"
        and config.get("trained_on_dispersion") is False
    ):
        method["dispersion"] = "D3"
    return method
