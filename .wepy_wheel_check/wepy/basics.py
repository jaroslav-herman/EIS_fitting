import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
from glob import glob
from galvani import MPRfile

# from hybdrt.models import DRT
from scipy.interpolate import interp1d


def data_average(df, variable_name="<I>/mA", val=1, f_min=0, f_max=2e4):
    freq_vec = df.loc[
        (df["cycle number"] == val) & (df["freq/Hz"] != 0), "freq/Hz"
    ].values
    var = df.loc[
        (df["cycle number"] == val) & (df["freq/Hz"] != 0), variable_name
    ].values
    mask = (freq_vec > f_min) & (freq_vec <= f_max)
    var = var[mask]
    avg = np.mean(var)
    return avg, var


def get_header_lines(filepath):
    with open(filepath, "r", encoding="latin1") as f:
        # Skip first line, read second
        f.readline()
        line2 = f.readline().strip()
    # line2 looks like: "Nb header lines : 76"
    # split by ":" and take the right part
    try:
        n_header = int(line2.split(":")[1])
    except Exception:
        raise ValueError(f"Could not parse header line: {line2}")
    return n_header


def read_header(filepath):
    """
    Reads the header lines from a file.

    Parameters:
    -----------
    filepath : str
        Path to the file to read.

    Returns:
    --------
    header : list of str
        List containing the header lines as strings, without trailing newline characters.
    """
    header_lines = get_header_lines(filepath)
    header = []
    with open(filepath, "r", encoding="latin1") as f:
        for _ in range(header_lines):
            header.append(f.readline().rstrip("\n"))
    return header


from natsort import natsorted


def load_files(
    folder,
    contains_string="",
    extension=".mpt",
    omit_string=None,
    natural_sort=False,
    mode="all",
):
    """
    Load files from a specified folder that contain specific substrings
    and have a certain file extension.

    Parameters:
    -----------
    folder : str
        Path to the folder where files will be searched.
    contains_string : str or list of str, optional
        Substring(s) used to filter filenames. Default is '' (matches all files).
    extension : str, optional
        File extension to filter by. Default is '.mpt'. Use 'all' to match all extensions.
    omit_string : str or list of str or None, optional
        Substring(s) that if found in a filename will cause it to be omitted. Default is None.
        If a list is provided, files containing any of the substrings are omitted.
    natural_sort : bool, optional
        If True, sort the resulting list naturally. Default is False.
    mode : str, optional
        Matching mode for contains_string when a list is provided:
        - 'all': filename must contain all substrings (default)
        - 'any': filename must contain at least one substring

    Returns:
    --------
    list of str
        List of file paths matching the criteria.
    """
    if extension == "all":
        extension = ""
    pattern = os.path.join(folder, f"*{extension}")
    files = glob(pattern)

    if mode not in {"all", "any"}:
        raise ValueError("mode must be 'all' or 'any'.")

    # Ensure contains_string is a list
    if isinstance(contains_string, str):
        contains_strings = [contains_string] if contains_string != "" else []
    elif contains_string is None:
        contains_strings = []
    else:
        contains_strings = list(contains_string)

    # Filter files according to selected mode
    if contains_strings:
        if mode == "all":
            files = [f for f in files if all(s in f for s in contains_strings)]
        else:  # mode == "any"
            files = [f for f in files if any(s in f for s in contains_strings)]

    # Ensure omit_string is a list if provided
    if omit_string is not None:
        if isinstance(omit_string, str):
            omit_strings = [omit_string]
        else:
            omit_strings = omit_string
        files = [f for f in files if not any(s in f for s in omit_strings)]

    if natural_sort:
        files = natsorted(files)
    return files


def load_folders(
    folder, contains_string="", omit_string=None, natural_sort=False, mode="all"
):
    """
    Load folders from a specified folder that contain specific substrings
    and optionally omit folders containing certain substrings.
    Parameters:
    -----------
    folder : str
        Path to the folder where folders will be searched.
    contains_string : str or list of str, optional
        Substring(s) used to filter folder names. Default is '' (matches all folders).
    omit_string : str or list of str or None, optional
        Substring(s) that if found in a folder name will cause it to be omitted. Default is None.
        If a list is provided, folders containing any of the substrings are omitted.
    natural_sort : bool, optional
        If True, sort the resulting list naturally. Default is False.
    mode : str, optional
        Matching mode for contains_string when a list is provided:
        - 'all': folder must contain all substrings (default)
        - 'any': folder must contain at least one substring

    Returns:
    --------
    list of str
        List of folder paths matching the criteria.
    """
    try:
        folders = [
            os.path.join(folder, name)
            for name in os.listdir(folder)
            if os.path.isdir(os.path.join(folder, name))
        ]
    except Exception as e:
        return f"Error accessing path: {e}"

    if mode not in {"all", "any"}:
        raise ValueError("mode must be 'all' or 'any'.")

    # Ensure contains_string is a list
    if isinstance(contains_string, str):
        contains_strings = [contains_string] if contains_string != "" else []
    elif contains_string is None:
        contains_strings = []
    else:
        contains_strings = list(contains_string)

    # Filter folders according to selected mode
    if contains_strings:
        if mode == "all":
            folders = [f for f in folders if all(s in f for s in contains_strings)]
        else:  # mode == "any"
            folders = [f for f in folders if any(s in f for s in contains_strings)]

    # Ensure omit_string is a list if provided
    if omit_string is not None:
        if isinstance(omit_string, str):
            omit_strings = [omit_string]
        else:
            omit_strings = omit_string
        folders = [f for f in folders if not any(s in f for s in omit_strings)]

    if natural_sort:
        folders = natsorted(folders)
    return folders


def get_colors(n=10, colormap="rainbow"):
    """
    Generate a list of colors sampled evenly from a matplotlib colormap.

    Parameters:
    -----------
    n : int, optional
        Number of colors to generate. Default is 10.
    colormap : str, optional
        Name of the matplotlib colormap to sample from. Default is 'rainbow'.

    Returns:
    --------
    colors : numpy.ndarray
        Array of RGBA colors sampled from the specified colormap.
    """
    cmap = plt.get_cmap(colormap)
    colors = cmap(np.linspace(0, 1, n))
    return colors


def _loop_labels(n_rows, loop_index):
    """Convert Galvani loop-start indices to zero-based row labels."""
    if n_rows == 0:
        return np.array([], dtype=int)
    if loop_index is None:
        return np.zeros(n_rows, dtype=int)

    starts = np.asarray(loop_index, dtype=int).reshape(-1)
    starts = starts[(starts >= 0) & (starts < n_rows)]
    starts = np.unique(np.concatenate(([0], starts)))
    labels = np.searchsorted(starts, np.arange(n_rows), side="right") - 1
    return labels.astype(int, copy=False)


def read_mpr(file, error_on_unknown_column=True):
    """Read a Bio-Logic ``.mpr`` file and add its zero-based ``loop`` column.

    Galvani stores loop starts as row indices in ``MPRfile.loop_index``.  The
    indices are converted into one integer loop label per data row.  Files
    without loop metadata receive ``loop == 0`` for every row.
    """
    mpr = MPRfile(os.fspath(file), error_on_unknown_column=error_on_unknown_column)
    data = pd.DataFrame(mpr.data)
    data["loop"] = _loop_labels(len(data), mpr.loop_index)
    return data


def read_file(file, skiprows=None, delimiter="\t", encoding="latin1", **kwargs):
    """
    Reads a data file into a pandas DataFrame.

    Parameters:
    -----------
    file : str
        Path to the file to read.
    skiprows : int or None, optional
        Number of rows to skip at the start of the file. If None, automatically
        determined by the header lines in the file. Default is None.
    delimiter : str, optional
        Delimiter used in the data file. Default is tab ('\t').
    encoding : str, optional
        File encoding. Default is 'latin1'.
    **kwargs :
        Additional keyword arguments passed to pandas.read_csv.

    Returns:
    --------
    pd.DataFrame
        Data read from the file.
    """
    if os.fspath(file).lower().endswith(".mpr"):
        # MPR files are binary Bio-Logic files and are read with the
        # MPR-specific reader so loop metadata is retained.
        return read_mpr(
            file,
            error_on_unknown_column=kwargs.pop("error_on_unknown_column", True),
        )

    if skiprows is None:
        skiprows = get_header_lines(file) - 1
    with open(file, encoding=encoding) as f:
        df = pd.read_csv(f, skiprows=skiprows, delimiter=delimiter, **kwargs)
    return df


def read_mpt_dataframe(file, **kwargs):
    """Read a Bio-Logic text export for the EIS Fitting compatibility API.

    Returns the dataframe together with header metadata and the technique
    label expected by the application-level import service.
    """
    dataframe = read_file(file, **kwargs)
    return dataframe, {}, "PEIS"


def read_file_safe(
    file,
    skiprows=None,
    delimiter="\t",
    encoding="latin1",
    *,
    on_error="skip",
    warn=True,
    **kwargs,
):
    """Read a measurement file and return ``None`` when it contains no data.

    Empty Bio-Logic exports can be zero-byte files, header-only files, or
    files that the reader cannot decode.  This helper is intended for batch
    processing: it skips those files by default while preserving the normal
    :func:`read_file` behavior for valid files.

    Parameters are the same as :func:`read_file`.

    Parameters
    ----------
    on_error : {"skip", "raise"}
        Skip unreadable files and return ``None`` (default), or re-raise the
        original exception.
    warn : bool
        Emit a warning when a file is skipped.
    """
    if on_error not in {"skip", "raise"}:
        raise ValueError("on_error must be 'skip' or 'raise'")

    path = os.fspath(file)
    reason = None
    try:
        if not os.path.isfile(path):
            reason = "file does not exist"
        elif os.path.getsize(path) == 0:
            reason = "file is empty"
        else:
            df = read_file(
                path,
                skiprows=skiprows,
                delimiter=delimiter,
                encoding=encoding,
                **kwargs,
            )
            if df is None or df.empty or df.dropna(how="all").empty:
                reason = "no data rows were saved"
            else:
                return df
    except Exception:
        if on_error == "raise":
            raise
        reason = "file could not be read"

    if warn:
        warnings.warn(f"Skipping {path!r}: {reason}.", UserWarning, stacklevel=2)
    return None


def get_sample_number(path):
    """
    Extracts the sample number from the last folder in the given path containing the sample number at its beginning.

    Parameters:
    -----------
    path : str
        Path to the file or folder.

    Returns:
    --------
    int
        The extracted sample number.

    Raises:
    -------
    ValueError
        If a sample number cannot be found in any folder of the path.
    """
    path = os.path.normpath(path)
    strings = path.split(os.sep)

    # If path includes a file (has extension), ignore last element (filename)
    end_idx = -2 if "." in strings[-1] else -1

    i = end_idx
    while abs(i) <= len(strings):
        folder_name = strings[i]
        # print(folder_name[:3])
        # Try to parse the first 3 characters as an integer sample number
        if len(folder_name) >= 3:
            try:

                number = int(folder_name[:3])
                break

            except ValueError:
                raise ValueError("No valid sample number found in path folders.")

        i -= 1
    return number


def get_sample_name(sample_number, sample_log="sample_log.csv"):
    """Return the sample name recorded for *sample_number*.

    The log must contain a ``Sample Name`` column.  The sample-number column
    may be named ``Sample Number``, ``Sample ID``, ``Sample`` or ``ID``; if
    none of those is present, all other columns are searched for an exact
    sample-number match.
    """
    log = pd.read_csv(sample_log)
    if "Sample Name" not in log.columns:
        raise KeyError(
            f"{sample_log!r} must contain a 'Sample Name' column; "
            f"found {list(log.columns)!r}"
        )

    target = str(sample_number).strip()
    normalized_columns = {
        column: "".join(ch for ch in str(column).lower() if ch.isalnum())
        for column in log.columns
    }
    preferred = [
        column
        for column, normalized in normalized_columns.items()
        if column != "Sample Name"
        and normalized in {"samplenumber", "sampleid", "sample", "id", "number"}
    ]
    search_columns = preferred or [
        column for column in log.columns if column != "Sample Name"
    ]

    for column in search_columns:
        values = log[column].astype("string").str.strip()
        matches = values == target
        if not matches.any():
            try:
                matches = pd.to_numeric(log[column], errors="coerce") == float(
                    sample_number
                )
            except (TypeError, ValueError):
                pass
        if matches.any():
            name = log.loc[matches, "Sample Name"].iloc[0]
            if pd.isna(name) or not str(name).strip():
                raise ValueError(
                    f"Sample {sample_number!r} has an empty 'Sample Name' in {sample_log!r}"
                )
            return str(name).strip()

    raise KeyError(
        f"Sample {sample_number!r} was not found in {sample_log!r}"
    )


def get_day(file):
    i = 0
    if "day" in file:
        start = file.find("day")

        if file[start + 3].isnumeric() == False:
            i = 1
        else:
            pass
        if file[start + i + 4].isnumeric():
            day = file[start : start + i + 5]
        else:
            day = file[start : start + i + 4]
    elif "Day" in file:
        start = file.find("Day")
        if file[start + 3].isnumeric() == False:
            i = 1
        else:
            pass
        if file[start + i + 4].isnumeric():
            day = file[start : start + i + 5]
        else:
            day = file[start : start + i + 4]
    else:
        raise Exception("No 'day' or 'Day' in file name.")
    return int(day[3 + i :])


def get_procedure(file):
    i = 0
    if "rocedur" in file:
        start = file.find("rocedur")

        if file[start + 8].isnumeric() == False:
            i = 1
            print(file[start + 8])
        else:
            pass
        if file[start + i + 9].isnumeric():
            procedure = file[start - 1 : start + i + 10]
        else:
            procedure = file[start - 1 : start + i + 9]
    else:
        raise Exception("No 'procedure' or 'Procedure' in file name.")
    return int(procedure[9 + i :])


def files_without_ext(files):
    if isinstance(files, str):
        files = [files]
    files_new = []
    for file in files:
        if file.find(".") == -1:
            start = 2
        else:
            start = 6
        for i in range(start, len(file)):
            if file[-i].isnumeric() and file[-i - 1].isnumeric():
                files_new.append(file[: -i - 2])
                break
            if i > 15:
                files_new.append(file)
                raise Exception("Cannot remove EC-Lab extension. Check the file name.")
                break
    return files_new


def files_with_ext(files, technique="SV"):
    if isinstance(files, str):
        files = [files]
    files_new = []
    for file in files:
        file = glob(os.path.join(file + "*" + technique + "*" + ".mpt"))[0]
        files_new.append(file)
    return files_new


def extend_cycles(cycles, series):
    series = np.array(series)
    cycles_new = []
    for cycle in cycles:
        for j, serie in enumerate(series):
            if cycle in serie:
                i = list(serie).index(cycle)
                for cycle_new in series[:, i]:
                    cycles_new.append(int(cycle_new))
    return cycles_new
