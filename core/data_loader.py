"""File loading and basic dataframe cleanup for VizCreate."""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO, Protocol

import pandas as pd


class UploadedFileLike(Protocol):
    """Minimal interface used by Streamlit uploaded files."""

    name: str

    def read(self, size: int = -1) -> bytes: ...

    def seek(self, offset: int, whence: int = 0) -> int: ...


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with whitespace-normalized, string column names.

    Newlines, tabs, and repeated spaces are collapsed into one space. Leading
    and trailing whitespace is removed. Data values are left unchanged.
    """
    cleaned = df.copy()
    cleaned.columns = (
        cleaned.columns
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    return cleaned


def load_dataframe(uploaded_file: UploadedFileLike | BinaryIO) -> pd.DataFrame:
    """Read a Streamlit-uploaded CSV or Excel file and clean its headers.

    Supported extensions are .csv, .xlsx, and .xls. A ValueError is raised for
    unsupported file types or empty datasets so the Streamlit layer can show a
    clear user-facing message.
    """
    file_name = getattr(uploaded_file, "name", "")
    suffix = Path(file_name).suffix.lower()

    try:
        if suffix == ".csv":
            df = pd.read_csv(uploaded_file)
        elif suffix in {".xlsx", ".xls"}:
            df = pd.read_excel(uploaded_file)
        else:
            raise ValueError(
                "Unsupported file type. Upload a CSV or Excel file "
                "(.csv, .xlsx, or .xls)."
            )
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"VizCreate could not read the uploaded file: {exc}") from exc

    if df.empty and len(df.columns) == 0:
        raise ValueError("The uploaded file does not contain a readable table.")

    return clean_column_names(df)
