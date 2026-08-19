"""Extract RelaxIS 3 (SQLite) impedance data into ML-friendly files."""
from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_FORMAT = "eis-fitting-project"
PROJECT_VERSION = 4


TABLES = ("Properties", "Projects", "Files", "Datapoints", "Fitparameters", "FileInformation")


def connect_read_only(path: Path) -> sqlite3.Connection:
    """Open a database without creating a journal, lock, or write capability."""
    uri = "file:" + str(path.resolve()).replace("\\", "/") + "?mode=ro&immutable=1"
    return sqlite3.connect(uri, uri=True)


def table_names(con: sqlite3.Connection) -> list[str]:
    return [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]


def read_table(con: sqlite3.Connection, table: str, columns: Iterable[str]) -> pd.DataFrame:
    available = {r[1] for r in con.execute(f'PRAGMA table_info("{table}")')}
    selected = [c for c in columns if c in available]
    if not selected:
        return pd.DataFrame()
    quoted = ", ".join('"' + c.replace('"', '""') + '"' for c in selected)
    return pd.read_sql_query(f'SELECT {quoted} FROM "{table}"', con)


def rename_existing(df: pd.DataFrame, mapping: dict[str, str], columns: list[str]) -> pd.DataFrame:
    df = df.rename(columns=mapping)
    for col in columns:
        if col not in df:
            df[col] = pd.Series(dtype="object")
    return df[columns]


def extract_database(input_file: str | Path, output_dir: str | Path) -> dict:
    source = Path(input_file)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    con = connect_read_only(source)
    try:
        existing = set(table_names(con))
        if "Files" not in existing or "Datapoints" not in existing:
            raise ValueError("Not a supported RelaxIS database: Files and Datapoints tables are required")

        props = read_table(con, "Properties", ["name", "value", "property", "key"])
        projects = read_table(con, "Projects", ["ID", "name", "description", "value", "project"])
        project_info = {"source_file": str(source), "properties": props.to_dict(orient="records"),
                        "projects": projects.to_dict(orient="records")}
        (destination / "project_info.json").write_text(json.dumps(project_info, indent=2, default=str), encoding="utf-8")

        files = read_table(con, "Files", ["ID", "lastweightmode", "lasttransferfunction", "lowfreqlimit", "highfreqlimit"])
        fits = rename_existing(files, {"ID":"file_id", "lastweightmode":"weighting_mode", "lasttransferfunction":"circuit_topology", "lowfreqlimit":"active_low_freq", "highfreqlimit":"active_high_freq"}, ["file_id", "weighting_mode", "circuit_topology", "active_low_freq", "active_high_freq"])
        points = read_table(con, "Datapoints", ["file_id", "frequency", "zreal", "zimag", "active", "zrealfit", "zimagfit"])
        spectra = rename_existing(points, {}, ["file_id", "frequency", "zreal", "zimag", "active", "zrealfit", "zimagfit"])
        if not spectra.empty:
            spectra["active"] = spectra["active"].fillna(0).astype(bool)
        else:
            spectra["active"] = pd.Series(dtype=bool)

        params = read_table(con, "Fitparameters", ["file_id", "pindex", "name", "value", "error", "lowerlimit", "upperlimit", "fixed", "isglobal", "fitted"])
        if "fitted" in params and not params.empty:
            params = params[params["fitted"].fillna(0).astype(bool)].copy()
        parameters = rename_existing(params, {"name":"parameter", "lowerlimit":"lower_limit", "upperlimit":"upper_limit"}, ["file_id", "pindex", "parameter", "value", "error", "lower_limit", "upper_limit", "fixed", "isglobal"])
        if "fixed" in parameters:
            parameters["fixed"] = parameters["fixed"].fillna(0).astype(bool)

        info = read_table(con, "FileInformation", ["file_id", "name", "value", "unit", "type"])
        metadata = rename_existing(info, {"name":"metadata_key", "value":"metadata_value"}, ["file_id", "metadata_key", "metadata_value", "unit", "type"])
        # RelaxIS stores numbers, paths, and free text in the same value column.
        # A nullable string column keeps that heterogeneous metadata Parquet-safe.
        if "metadata_value" in metadata:
            metadata["metadata_value"] = metadata["metadata_value"].astype("string")
        for frame, name in ((fits,"fits.parquet"), (spectra,"spectra.parquet"), (parameters,"parameters.parquet"), (metadata,"metadata.parquet")):
            frame.to_parquet(destination / name, index=False)
        return stats(con, source, fits, spectra)
    finally:
        con.close()


def stats(con, source: Path, fits: pd.DataFrame | None = None, spectra: pd.DataFrame | None = None) -> dict:
    if fits is None: fits = read_table(con, "Files", ["ID", "lasttransferfunction"])
    if spectra is None: spectra = read_table(con, "Datapoints", ["file_id", "frequency"])
    return {"input_file": str(source), "spectra": int(fits["ID"].nunique() if "ID" in fits else fits["file_id"].nunique()), "points": int(len(spectra)), "tables": ",".join(table_names(con))}


def inspect_database(path: Path) -> None:
    con = connect_read_only(path)
    try:
        names = table_names(con)
        print(f"RelaxIS database: {path}")
        print(f"SQLite version: {sqlite3.sqlite_version}; user_version: {con.execute('PRAGMA user_version').fetchone()[0]}")
        print("Tables: " + ", ".join(names))
        if "Files" in names:
            f = read_table(con, "Files", ["ID", "lasttransferfunction"])
            print(f"Total spectra: {len(f)}")
            if "lasttransferfunction" in f:
                print("Circuit models: " + ", ".join(map(str, f["lasttransferfunction"].dropna().unique())) )
        if "Datapoints" in names:
            d = read_table(con, "Datapoints", ["file_id", "active"])
            print(f"Total points: {len(d)}")
            if "file_id" in d:
                print("Point count breakdown:")
                print(d.groupby("file_id").size().to_string())
    finally: con.close()


def _canonical_datasource(value: object) -> str:
    """Normalize RelaxIS copies so model copies match their unassigned source."""
    value = str(value or "").strip().replace("\\", "/")
    while value.lower().startswith("copy of "):
        value = value[8:].strip()
    return value.casefold()


def _relaxis_circuit(model: object) -> str | None:
    """Translate a compatible RelaxIS topology to impedance.py syntax.

    RelaxIS stores compact labels such as ``R-I-(R)(P)-(R)(P)``.  The
    parser deliberately returns ``None`` for unknown topologies so an
    incorrect EEC is never assigned silently.
    """
    text = re.sub(r"\s+", "", str(model or "").strip())
    if not text or text.casefold() in {"impedance", "unassignedspectra"}:
        return None
    if re.fullmatch(r"R\d+(?:-L\d+)?(?:-p\(R\d+,CPE\d+\))*", text, re.I):
        return text
    match = re.fullmatch(r"(R(?:-I)?)(?:-\(R\)\(P\))*", text, re.I)
    if not match:
        return None
    base = "R0-L0" if match.group(1).casefold() == "r-i" else "R0"
    groups = len(re.findall(r"\(R\)\(P\)", text, re.I))
    return base + ("-" if groups else "") + "-".join(
        f"p(R{i},CPE{i})" for i in range(1, groups + 1)
    )


def _finite_relaxis_limit(value: object) -> float | None:
    """Convert RelaxIS limits, treating huge/sentinel values as unset."""
    number = _json_number(value)
    if number is None or number <= 0 or abs(number) >= 1e100:
        return None
    return number


def _frequency_window(frame: pd.DataFrame, instance: dict) -> list[float]:
    measured = (float(frame["freq_hz"].min()), float(frame["freq_hz"].max()))
    low = _finite_relaxis_limit(instance.get("lowfreqlimit"))
    high = _finite_relaxis_limit(instance.get("highfreqlimit"))
    if low is None:
        low = measured[0]
    if high is None:
        high = measured[1]
    if low > high:
        return list(measured)
    return [low, high]


def _parameter_name(relaxis_name: object, circuit: str) -> tuple[str, str]:
    name = str(relaxis_name or "").strip()
    resistance = re.fullmatch(r"Resistance\s+(\d+)", name, re.I)
    if resistance:
        index = int(resistance.group(1)) - 1
        return f"R{index}", "Ohm"
    inductance = re.fullmatch(r"Inductance\s+(\d+)", name, re.I)
    if inductance:
        return f"L{int(inductance.group(1)) - 1}", "H"
    q = re.fullmatch(r"CPE\s+Q\s+(\d+)", name, re.I)
    if q:
        return f"CPE{int(q.group(1))}_0", "Ohm^-1 sec^a"
    alpha = re.fullmatch(r"CPE\s+Alpha\s+(\d+)", name, re.I)
    if alpha:
        return f"CPE{int(alpha.group(1))}_1", ""
    return name, ""


def _json_number(value: object, default: float | None = None) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def discover_metadata_keys(eis3_path: str | Path) -> list[str]:
    """Return available FileInformation keys, excluding RelaxIS-only flags."""
    con = connect_read_only(Path(eis3_path))
    try:
        if "FileInformation" not in table_names(con):
            return []
        rows = pd.read_sql_query(
            'SELECT DISTINCT name FROM "FileInformation" WHERE name IS NOT NULL ORDER BY name',
            con,
        )
        return [str(name) for name in rows["name"] if str(name) != "IsEpsOnlyData"]
    finally:
        con.close()


def _eisfit_metadata(values: dict[str, object], metadata_mapping: dict[str, str] | None = None) -> dict[str, object]:
    """Apply user-approved metadata inclusion and output-name mappings."""
    mapped: dict[str, object] = {}
    for key, value in values.items():
        if key == "IsEpsOnlyData":
            continue
        if metadata_mapping is not None:
            if key not in metadata_mapping:
                continue
            output_key = metadata_mapping[key]
        else:
            output_key = {"DCVoltage": "Ecell_V", "Current": "I_mA"}.get(key, key)
        if output_key and output_key != "IsEpsOnlyData":
            mapped[output_key] = value
    return mapped


def _strip_eisfit_exclusions(value):
    """Recursively remove fields that must never enter an eisfit project."""
    if isinstance(value, dict):
        return {
            key: _strip_eisfit_exclusions(item)
            for key, item in value.items()
            if key != "IsEpsOnlyData"
        }
    if isinstance(value, list):
        return [_strip_eisfit_exclusions(item) for item in value]
    return value


def _conflict_choice(handler, datasource: str, instances: list[dict]) -> list[dict]:
    if handler is None:
        if not sys.stdin.isatty():
            return instances
        answer = input(f"Datasource {datasource!r} has multiple EEC models. Export all? [Y/n] ").strip().lower()
        return instances if answer in ("", "y", "yes", "a", "all") else [instances[0]]
    choice = handler(datasource, instances)
    if choice in (None, "all", "ALL", True):
        return instances
    if isinstance(choice, dict):
        if choice.get("all"):
            return instances
        index = choice.get("index")
        if isinstance(index, int) and 0 <= index < len(instances):
            selected = dict(instances[index])
        else:
            selected = dict(instances[0])
        if choice.get("model"):
            selected["model"] = choice["model"]
        if choice.get("circuit"):
            selected["circuit"] = choice["circuit"]
        return [selected]
    if isinstance(choice, int):
        return [instances[choice]]
    if isinstance(choice, (list, tuple)):
        if all(isinstance(item, int) for item in choice):
            return [instances[item] for item in choice]
        return list(choice)
    return [next(item for item in instances if item["model"] == choice)]


def export_to_eisfit_json(eis3_path: str | Path, output_dir: str | Path, model_conflict_handler=None,
                          metadata_mapping: dict[str, str] | None = None,
                          unmapped_model_handler=None) -> Path:
    """Convert a RelaxIS database into the built-in EIS-fitting project format."""
    source = Path(eis3_path).resolve()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    con = connect_read_only(source)
    try:
        files = read_table(con, "Files", ["ID", "groupname", "datasource", "fitted", "lasttransferfunction", "lowfreqlimit", "highfreqlimit"])
        files = files.sort_values("ID", kind="mergesort") if "ID" in files else files
        points = read_table(con, "Datapoints", ["ID", "file_id", "frequency", "zreal", "zimag", "active"])
        if {"file_id", "ID"}.issubset(points.columns):
            points = points.sort_values(["file_id", "ID"], kind="mergesort")
        parameters = read_table(con, "Fitparameters", ["file_id", "pindex", "name", "fixed", "value", "error", "lowerlimit", "upperlimit"])
        if {"file_id", "pindex"}.issubset(parameters.columns):
            parameters = parameters.sort_values(["file_id", "pindex"], kind="mergesort")
        info = read_table(con, "FileInformation", ["file_id", "name", "value"])
    finally:
        con.close()

    grouped: dict[str, dict] = {}
    for row in files.to_dict("records"):
        key = _canonical_datasource(row["datasource"])
        if not key:
            continue
        group = str(row.get("groupname") or "").strip()
        assigned = group.casefold() != "unassigned spectra"
        topology = row.get("lasttransferfunction")
        if not topology or str(topology).casefold() == "impedance":
            topology = group
        instance = {"file_id": int(row["ID"]), "datasource": row["datasource"], "model": group, "assigned": assigned,
                    "circuit": _relaxis_circuit(topology) if assigned else "R0-L0-p(R1,CPE1)", "fitted": bool(row.get("fitted")),
                    "lowfreqlimit": row.get("lowfreqlimit"), "highfreqlimit": row.get("highfreqlimit")}
        grouped.setdefault(key, {"display": str(row["datasource"]), "assigned": [], "unassigned": []})["assigned" if assigned else "unassigned"].append(instance)

    selected: list[dict] = []
    for group in grouped.values():
        candidates = group["assigned"] or group["unassigned"]
        if group["assigned"] and len({item["model"] for item in group["assigned"]}) > 1:
            candidates = _conflict_choice(model_conflict_handler, group["display"], group["assigned"])
        selected.extend(candidates)

    info_by_file: dict[int, dict[str, object]] = {}
    for row in info.to_dict("records"):
        info_by_file.setdefault(int(row["file_id"]), {})[str(row["name"])] = row["value"]
    datasets = []
    for cycle_number, instance in enumerate(selected, 1):
        file_id = instance["file_id"]
        frame = points[points["file_id"] == file_id].copy()
        frame = frame[frame["frequency"].notna() & frame["zreal"].notna() & frame["zimag"].notna()]
        if frame.empty:
            continue
        if instance["circuit"] is None:
            if unmapped_model_handler is None:
                raise ValueError(
                    f"RelaxIS model {instance['model']!r} for datasource "
                    f"{instance['datasource']!r} cannot be mapped automatically"
                )
            mapped = unmapped_model_handler(instance["model"], instance)
            if isinstance(mapped, dict):
                instance = {**instance, **mapped}
            else:
                instance = {**instance, "circuit": mapped}
            if not instance.get("circuit"):
                raise ValueError(f"No EIS-fitting circuit was selected for {instance['model']!r}")
        metadata = {"source_name": instance["datasource"], "relaxis_group": instance["model"]}
        metadata.update(_eisfit_metadata(info_by_file.get(file_id, {}), metadata_mapping))
        frame["freq_hz"] = frame["frequency"].astype(float)
        frame["cycle_number"] = cycle_number
        frame["re_z_ohm"] = frame["zreal"].astype(float)
        frame["minus_im_z_ohm"] = -frame["zimag"].astype(float)
        frame["ewe_v"] = _json_number(info_by_file.get(file_id, {}).get("DCVoltage"), 0.0)
        frame["i_ma"] = _json_number(info_by_file.get(file_id, {}).get("Current"), 0.0)
        frame["time_s"] = _json_number(info_by_file.get(file_id, {}).get("Time"), None)
        # Keep every point-wise field aligned with the descending order used
        # by eis_services.load_cycle().
        frame = frame.sort_values("freq_hz", ascending=False, kind="mergesort").reset_index(drop=True)
        dataframe = frame[["freq_hz", "cycle_number", "re_z_ohm", "minus_im_z_ohm", "ewe_v", "i_ma", "time_s"]]
        params = []
        for row in parameters[parameters["file_id"] == file_id].to_dict("records"):
            param_name, unit = _parameter_name(row["name"], instance["circuit"])
            value = _json_number(row["value"], 0.0)
            error = _json_number(row["error"], None)
            params.append({"name": param_name, "unit": unit, "initial": value, "lower": _json_number(row["lowerlimit"], 0.0),
                          "upper": _json_number(row["upperlimit"], 1e12), "error_percent": (abs(error / value) * 100 if error is not None and value not in (None, 0) else None), "fixed": bool(row["fixed"])})
        state = {"format": PROJECT_FORMAT, "version": PROJECT_VERSION, "source_path": str(source), "circuit": instance["circuit"], "control": "working",
                 "active_cycle": cycle_number, "all_frequency_window": None, "default_parameters": params,
                 "cycles": {str(cycle_number): {"circuit": instance["circuit"], "potential_v": dataframe["ewe_v"].iloc[0], "current_ma": dataframe["i_ma"].iloc[0], "time_s": dataframe["time_s"].iloc[0],
                 "frequency_window": _frequency_window(frame, instance), "auto_max_frequency": False,
                 "manually_included": frame["active"].fillna(0).astype(bool).tolist(), "outliers": [False] * len(frame), "parameters": params, "fit_parameters": None,
                 "fit_frequency_hz": None, "fit_impedance": None, "fit_at_data_impedance": None, "ridge_tau_s": None, "ridge_gamma_ohm": None, "drt_label": None,
                 "saved_ridge_tau_s": None, "saved_ridge_gamma_ohm": None, "saved_ridge_included_mask": None, "saved_ridge_outlier_indices": None, "saved_ridge_parameters": [],
                 "saved_ridge_threshold": None, "saved_ridge_peak_count": None, "saved_ridge_ohmic_resistance": None, "saved_ridge_inductance": None, "saved_ridge_peak_parameters": [],
                 "saved_hybrid_tau_s": None, "saved_hybrid_gamma_ohm": None, "saved_hybrid_included_mask": None, "saved_hybrid_ohmic_resistance": None, "saved_hybrid_inductance": None, "saved_hybrid_peak_parameters": [], "custom_metadata": metadata}}}
        datasets.append({"dataset_id": f"relaxis::{cycle_number}::{file_id}", "state": state, "dataframe": dataframe.to_json(orient="split")})
    if not datasets:
        raise ValueError("No usable impedance spectra were found")
    payload = _strip_eisfit_exclusions(datasets[0]["state"] | {"datasets": datasets})
    output = destination / f"{source.stem}.eisfit.json"
    output.write_text(json.dumps(payload, indent=2, allow_nan=False, default=str), encoding="utf-8")
    return output


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input_file", type=Path)
    ap.add_argument("output_dir", type=Path, nargs="?", default=Path("relaxis_output"))
    ap.add_argument("--inspect", action="store_true", help="print diagnostics without writing output")
    args = ap.parse_args(argv)
    if args.input_file.is_file():
        if args.inspect: inspect_database(args.input_file); return 0
        print(json.dumps(extract_database(args.input_file, args.output_dir), indent=2)); return 0
    if not args.input_file.is_dir(): ap.error(f"Input does not exist: {args.input_file}")
    if args.inspect: 
        for p in args.input_file.rglob("*.eis3"): inspect_database(p)
        return 0
    rows = []
    for p in args.input_file.rglob("*.eis3"):
        rel = p.relative_to(args.input_file).with_suffix("")
        rows.append(extract_database(p, args.output_dir / rel))
    pd.DataFrame(rows).to_csv(args.output_dir / "batch_summary.csv", index=False)
    print(f"Processed {len(rows)} database(s)")
    return 0


if __name__ == "__main__": raise SystemExit(main())
