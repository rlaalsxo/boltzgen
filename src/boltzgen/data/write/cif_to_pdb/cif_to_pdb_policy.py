from __future__ import annotations

import re
from dataclasses import replace

from .cif_to_pdb_types import (
    AtomRecord,
    ConversionMode,
    FormattedAtomRecord,
    MAX_ATOM_SERIAL,
    MAX_RESSEQ,
    MIN_RESSEQ,
    PDB_CHAIN_IDS,
)


def apply_policy(records: list[AtomRecord], mode: ConversionMode, warnings: list[str]) -> list[FormattedAtomRecord]:
    records = _reduce_models(records, mode, warnings)
    records = _filter_altloc(records, mode, warnings)
    if not records:
        raise ValueError("No atoms remain after applying conversion policy.")

    chain_map = _build_chain_map(records, mode, warnings)
    residue_plan = _build_residue_plan(records, chain_map, mode, warnings)
    return _build_formatted_records(records, chain_map, residue_plan, mode, warnings)


def _reduce_models(records: list[AtomRecord], mode: ConversionMode, warnings: list[str]) -> list[AtomRecord]:
    models: list[int] = []
    seen = set()
    for record in records:
        if record.model_num not in seen:
            seen.add(record.model_num)
            models.append(record.model_num)

    if len(models) <= 1:
        return records
    if mode == "strict":
        raise ValueError("Strict mode does not support multi-model CIF input.")

    _warn(warnings, f"Reduced multi-model CIF to the first model ({models[0]}).")
    return [record for record in records if record.model_num == models[0]]


def _filter_altloc(records: list[AtomRecord], mode: ConversionMode, warnings: list[str]) -> list[AtomRecord]:
    if mode == "strict":
        for record in records:
            if len(record.altloc.strip()) > 1:
                raise ValueError("Strict mode requires altloc identifiers to fit in one character.")
        return records

    grouped: dict[tuple[str, str, str, str, str, str], list[AtomRecord]] = {}
    for record in records:
        key = (
            record.chain_id,
            record.resseq_raw,
            record.ins_code,
            record.resname,
            record.atom_name,
            record.record_type,
        )
        grouped.setdefault(key, []).append(record)

    pruned = 0
    selected: list[AtomRecord] = []
    for group in grouped.values():
        if len(group) == 1:
            selected.append(replace(group[0], altloc=""))
            continue

        winner = sorted(group, key=lambda item: (-item.occupancy, item.altloc != "", item.source_index))[0]
        selected.append(replace(winner, altloc=""))
        pruned += 1

    if pruned:
        _warn(warnings, f"Pruned alternate locations for {pruned} atom groups.")

    selected.sort(key=lambda item: item.source_index)
    return selected


def _build_chain_map(records: list[AtomRecord], mode: ConversionMode, warnings: list[str]) -> dict[str, str]:
    ordered_chain_ids: list[str] = []
    seen = set()
    for record in records:
        if record.chain_id not in seen:
            seen.add(record.chain_id)
            ordered_chain_ids.append(record.chain_id)

    if mode == "strict":
        invalid = [chain_id for chain_id in ordered_chain_ids if len(chain_id) != 1]
        if invalid:
            raise ValueError(f"Strict mode requires single-character chain IDs. Found: {invalid[0]!r}")
        return {chain_id: chain_id for chain_id in ordered_chain_ids}

    if len(ordered_chain_ids) > len(PDB_CHAIN_IDS):
        raise ValueError(
            f"Cannot map {len(ordered_chain_ids)} chains into single-character PDB chain IDs."
        )

    chain_map: dict[str, str] = {}
    used = set()
    for chain_id in ordered_chain_ids:
        if len(chain_id) == 1 and chain_id in PDB_CHAIN_IDS and chain_id not in used:
            chain_map[chain_id] = chain_id
            used.add(chain_id)

    available = [chain_id for chain_id in PDB_CHAIN_IDS if chain_id not in used]
    remapped = False
    for chain_id in ordered_chain_ids:
        if chain_id in chain_map:
            continue
        chain_map[chain_id] = available.pop(0)
        remapped = True

    if remapped:
        _warn(warnings, "Remapped chain IDs to fit single-character PDB fields.")

    return chain_map


def _build_residue_plan(
    records: list[AtomRecord],
    chain_map: dict[str, str],
    mode: ConversionMode,
    warnings: list[str],
) -> dict[tuple[str, str, str, str, str], int]:
    residue_plan: dict[tuple[str, str, str, str, str], int] = {}
    chain_should_renumber: dict[str, bool] = {}
    chain_next_number: dict[str, int] = {}
    wrapped = set()

    residue_keys = _iter_residue_keys(records, chain_map)
    for residue_key, record in residue_keys:
        mapped_chain = residue_key[0]
        parsed_resseq = _parse_resseq(record.resseq_raw)
        needs_renumber = parsed_resseq is None or parsed_resseq < MIN_RESSEQ or parsed_resseq > MAX_RESSEQ

        if mode == "strict":
            if needs_renumber:
                raise ValueError(
                    f"Strict mode requires residue IDs within PDB range. Found: {record.resseq_raw!r}"
                )
            residue_plan[residue_key] = parsed_resseq
            continue

        chain_should_renumber[mapped_chain] = chain_should_renumber.get(mapped_chain, False) or needs_renumber

    for residue_key, record in _iter_residue_keys(records, chain_map):
        mapped_chain = residue_key[0]
        parsed_resseq = _parse_resseq(record.resseq_raw)
        if not chain_should_renumber.get(mapped_chain, False) and parsed_resseq is not None:
            residue_plan[residue_key] = parsed_resseq
            continue

        next_number = chain_next_number.get(mapped_chain, 1)
        if next_number > MAX_RESSEQ:
            if mode == "best_effort":
                next_number = 1
                wrapped.add(mapped_chain)
            else:
                raise ValueError(
                    f"Compatible mode cannot renumber more than {MAX_RESSEQ} residues in one chain."
                )

        residue_plan[residue_key] = next_number
        chain_next_number[mapped_chain] = next_number + 1

    if any(chain_should_renumber.values()):
        _warn(warnings, "Renumbered residues that do not fit the PDB residue field.")
    if wrapped:
        _warn(warnings, "Wrapped residue numbering in best_effort mode.")

    return residue_plan


def _build_formatted_records(
    records: list[AtomRecord],
    chain_map: dict[str, str],
    residue_plan: dict[tuple[str, str, str, str, str], int],
    mode: ConversionMode,
    warnings: list[str],
) -> list[FormattedAtomRecord]:
    formatted: list[FormattedAtomRecord] = []
    next_serial = 1
    wrapped_serial = False

    for record in records:
        residue_key = (
            chain_map[record.chain_id],
            record.chain_id,
            record.resseq_raw,
            record.ins_code,
            record.resname,
        )
        serial = _resolve_serial(record.original_serial, next_serial, mode)
        if serial != next_serial:
            wrapped_serial = True
        next_serial += 1

        formatted.append(
            FormattedAtomRecord(
                source_index=record.source_index,
                record_type=record.record_type,
                serial=serial,
                atom_name=_normalize_field(record.atom_name, 4, "atom name", mode, warnings),
                altloc=_normalize_single_char(record.altloc, "altloc", mode),
                resname=_normalize_field(record.resname or "UNK", 3, "residue name", mode, warnings),
                chain_id=chain_map[record.chain_id],
                resseq=residue_plan[residue_key],
                ins_code=_normalize_single_char(record.ins_code, "insertion code", mode),
                x=record.x,
                y=record.y,
                z=record.z,
                occupancy=record.occupancy,
                bfactor=record.bfactor,
                element=_normalize_element(record.element, record.atom_name),
                charge=_normalize_charge(record.charge, mode, warnings),
            )
        )

    if wrapped_serial:
        _warn(warnings, "Wrapped atom serial numbers in best_effort mode.")

    return formatted


def _iter_residue_keys(records: list[AtomRecord], chain_map: dict[str, str]):
    seen = set()
    for record in records:
        residue_key = (
            chain_map[record.chain_id],
            record.chain_id,
            record.resseq_raw,
            record.ins_code,
            record.resname,
        )
        if residue_key in seen:
            continue
        seen.add(residue_key)
        yield residue_key, record


def _parse_resseq(value: str) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _resolve_serial(original_serial: int | None, next_serial: int, mode: ConversionMode) -> int:
    if mode == "strict":
        if original_serial is None or original_serial < 1 or original_serial > MAX_ATOM_SERIAL:
            raise ValueError(
                f"Strict mode requires atom serials within 1-{MAX_ATOM_SERIAL}. Found: {original_serial!r}"
            )
        return original_serial

    if next_serial > MAX_ATOM_SERIAL:
        if mode == "best_effort":
            return ((next_serial - 1) % MAX_ATOM_SERIAL) + 1
        raise ValueError(
            f"Compatible mode cannot represent more than {MAX_ATOM_SERIAL} atoms in a single PDB file."
        )

    return next_serial


def _normalize_field(
    value: str,
    width: int,
    label: str,
    mode: ConversionMode,
    warnings: list[str],
) -> str:
    value = value.strip()
    if len(value) <= width:
        return value
    if mode == "strict":
        raise ValueError(f"Strict mode does not allow {label}s longer than {width} characters: {value!r}")
    _warn(warnings, f"Truncated {label}s longer than {width} characters.")
    return value[:width]


def _normalize_single_char(value: str, label: str, mode: ConversionMode) -> str:
    value = value.strip()
    if len(value) <= 1:
        return value
    if mode == "strict":
        raise ValueError(f"Strict mode does not allow {label}s longer than 1 character: {value!r}")
    return value[:1]


def _normalize_element(element: str, atom_name: str) -> str:
    element = element.strip()
    if element:
        return element.upper() if len(element) == 1 else f"{element[0].upper()}{element[1].lower()}"

    letters = re.sub(r"[^A-Za-z]", "", atom_name)
    if not letters:
        return "X"
    return letters.upper() if len(letters) == 1 else f"{letters[0].upper()}{letters[1].lower()}"


def _normalize_charge(charge: str, mode: ConversionMode, warnings: list[str]) -> str:
    charge = charge.strip()
    if not charge:
        return ""
    if re.fullmatch(r"[1-9][+-]", charge):
        return charge
    if re.fullmatch(r"[+-][1-9]", charge):
        return f"{charge[1]}{charge[0]}"
    if re.fullmatch(r"[+-]?\d+", charge):
        value = int(charge)
        if value == 0:
            return ""
        magnitude = abs(value)
        if magnitude <= 9:
            return f"{magnitude}{'+' if value > 0 else '-'}"
        if mode == "strict":
            raise ValueError(f"Strict mode does not allow charge values outside PDB field width: {charge!r}")
        _warn(warnings, "Dropped charges that do not fit the PDB charge field.")
        return ""

    if mode == "strict":
        raise ValueError(f"Strict mode does not allow unsupported charge values: {charge!r}")
    _warn(warnings, "Dropped unsupported formal charge values.")
    return ""


def _warn(warnings: list[str], message: str):
    if message not in warnings:
        warnings.append(message)
