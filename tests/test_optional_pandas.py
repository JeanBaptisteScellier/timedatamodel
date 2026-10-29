"""pandas is an optional dependency (the ``[pandas]`` extra).

The package must import and work without it; only the pandas interop
methods may require it, and they must say how to install it.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from datetime import UTC, datetime

import polars as pl
import pytest


def test_import_without_pandas():
    """``import timedatamodel`` must not import pandas."""
    code = textwrap.dedent(
        """
        import sys
        sys.modules["pandas"] = None  # make any `import pandas` fail
        import timedatamodel
        from timedatamodel import TimeSeries  # noqa: F401
        from timedatamodel.timeseriestable import TimeSeriesTable  # noqa: F401
        assert "pandas" not in [m for m, v in sys.modules.items() if v is not None]
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code], check=False, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_polars_roundtrip_without_pandas(monkeypatch):
    monkeypatch.setitem(sys.modules, "pandas", None)
    from timedatamodel import Frequency, TimeSeries

    df = pl.DataFrame(
        {
            "valid_time": [datetime(2024, 1, 1, h, tzinfo=UTC) for h in range(3)],
            "value": [1.0, 2.0, 3.0],
        }
    )
    ts = TimeSeries.from_polars(df, frequency=Frequency.PT1H, name="x")
    assert ts.to_polars().height == 3


@pytest.mark.parametrize("method", ["from_pandas", "to_pandas"])
def test_pandas_methods_raise_helpful_error(monkeypatch, method):
    monkeypatch.setitem(sys.modules, "pandas", None)
    from timedatamodel import Frequency, TimeSeries

    df = pl.DataFrame(
        {
            "valid_time": [datetime(2024, 1, 1, tzinfo=UTC)],
            "value": [1.0],
        }
    )
    ts = TimeSeries.from_polars(df, frequency=Frequency.PT1H, name="x")
    with pytest.raises(ImportError, match=r"timedatamodel\[pandas\]"):
        if method == "from_pandas":
            TimeSeries.from_pandas(object(), name="x", frequency=Frequency.PT1H)
        else:
            ts.to_pandas()
