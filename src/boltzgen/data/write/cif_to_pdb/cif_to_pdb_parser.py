from __future__ import annotations

import re
from collections.abc import Iterable

from .cif_to_pdb_types import (
    AssemblyGen,
    AssemblyInfo,
    AssemblyOperation,
    AtomRecord,
    ConversionStats,
    CrystInfo,
    NULL_TOKENS,
)


def normalize_mode(mode: str):
    from .cif_to_pdb_types import SUPPORTED_MODES

    normalized = (mode or "compatible").strip().lower()
    if normalized not in SUPPORTED_MODES:
        raise ValueError(
            f"Unsupported conversion mode: {mode}. Supported modes: strict, compatible, best_effort."
        )
    return normalized


def read_gemmi_block(cif_content: bytes):
    import gemmi
    import os
    import tempfile

    if hasattr(gemmi.cif, "read_string"):
        return gemmi.cif.read_string(cif_content.decode("utf-8", errors="replace")).sole_block()

    tmp_cif = tempfile.NamedTemporaryFile(suffix=".cif", delete=False)
    try:
        tmp_cif.write(cif_content)
        tmp_cif.close()
        return gemmi.cif.read(tmp_cif.name).sole_block()
    finally:
        if os.path.exists(tmp_cif.name):
            os.unlink(tmp_cif.name)


def extract_atom_records(atom_site) -> list[AtomRecord]:
    if not atom_site:
        return []

    first_key = next(iter(atom_site.keys()), None)
    if first_key is None:
        return []

    row_count = len(atom_site[first_key])
    records: list[AtomRecord] = []

    for index in range(row_count):
        record_type = _get_value(atom_site, ("group_PDB",), index, "ATOM").upper()
        if record_type not in {"ATOM", "HETATM"}:
            continue

        atom_name = _get_value(atom_site, ("label_atom_id", "auth_atom_id"), index)
        if not atom_name:
            atom_name = _get_value(atom_site, ("type_symbol",), index, "X")

        resname = _get_value(atom_site, ("label_comp_id", "auth_comp_id"), index, "UNK")
        chain_id = _get_value(atom_site, ("auth_asym_id", "label_asym_id"), index, "_")
        resseq_raw = _get_value(atom_site, ("auth_seq_id", "label_seq_id"), index) or "1"
        altloc = _clean_token(_get_value(atom_site, ("label_alt_id",), index))
        ins_code = _clean_token(_get_value(atom_site, ("pdbx_PDB_ins_code",), index))
        charge = _clean_token(_get_value(atom_site, ("pdbx_formal_charge",), index))
        model_num = _parse_int(_get_value(atom_site, ("pdbx_PDB_model_num",), index, "1")) or 1
        original_serial = _parse_int(_get_value(atom_site, ("id",), index))

        records.append(
            AtomRecord(
                source_index=index,
                record_type=record_type,
                original_serial=original_serial,
                atom_name=atom_name.replace("'", "").replace('"', ""),
                altloc=altloc,
                resname=resname,
                chain_id=chain_id,
                resseq_raw=resseq_raw,
                ins_code=ins_code,
                x=_parse_float(_get_value(atom_site, ("Cartn_x",), index)),
                y=_parse_float(_get_value(atom_site, ("Cartn_y",), index)),
                z=_parse_float(_get_value(atom_site, ("Cartn_z",), index)),
                occupancy=_parse_float(_get_value(atom_site, ("occupancy",), index), 1.0),
                bfactor=_parse_float(_get_value(atom_site, ("B_iso_or_equiv",), index), 0.0),
                element=_get_value(atom_site, ("type_symbol",), index, "X"),
                charge=charge,
                model_num=model_num,
            )
        )

    return records


def extract_cryst_info(block) -> CrystInfo | None:
    cell = block.find("_cell.", ["length_a", "length_b", "length_c", "angle_alpha", "angle_beta", "angle_gamma"])
    if not cell or len(cell) == 0:
        return None

    row = cell[0]
    try:
        a = float(row[0])
        b = float(row[1])
        c = float(row[2])
        alpha = float(row[3])
        beta = float(row[4])
        gamma = float(row[5])
    except (ValueError, TypeError):
        return None

    sg_raw = _get_block_value(block, "_symmetry.space_group_name_H-M") or \
             _get_block_value(block, "_space_group.name_H-M_alt") or "P 1"
    space_group = sg_raw.strip().strip("'\"")

    z_raw = _get_block_value(block, "_cell.Z_PDB") or _get_block_value(block, "_cell.formula_units_Z") or "1"
    try:
        z_value = int(z_raw.strip())
    except (ValueError, TypeError):
        z_value = 1

    return CrystInfo(a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma, space_group=space_group, z_value=z_value)


def extract_assembly_info(block) -> list[AssemblyInfo]:
    operations = _extract_operations(block)
    if not operations:
        return []

    assemblies_meta = _extract_assembly_meta(block)
    gens_by_assembly = _extract_assembly_gens(block)

    result: list[AssemblyInfo] = []
    for assembly_id, (details, oligo) in assemblies_meta.items():
        gens = gens_by_assembly.get(assembly_id, [])
        result.append(AssemblyInfo(
            assembly_id=assembly_id,
            details=details,
            oligomeric_details=oligo,
            gens=gens,
            operations=operations,
        ))

    return result


def _extract_operations(block) -> dict[str, AssemblyOperation]:
    table = block.find(
        "_pdbx_struct_oper_list.",
        ["id", "matrix[1][1]", "matrix[1][2]", "matrix[1][3]",
         "matrix[2][1]", "matrix[2][2]", "matrix[2][3]",
         "matrix[3][1]", "matrix[3][2]", "matrix[3][3]",
         "vector[1]", "vector[2]", "vector[3]"],
    )
    if not table:
        return {}

    ops: dict[str, AssemblyOperation] = {}
    for row in table:
        oper_id = str(row[0]).strip()
        try:
            matrix = [
                [float(row[1]), float(row[2]), float(row[3])],
                [float(row[4]), float(row[5]), float(row[6])],
                [float(row[7]), float(row[8]), float(row[9])],
            ]
            vector = [float(row[10]), float(row[11]), float(row[12])]
        except (ValueError, TypeError):
            continue
        ops[oper_id] = AssemblyOperation(oper_id=oper_id, matrix=matrix, vector=vector)

    return ops


def _extract_assembly_meta(block) -> dict[str, tuple[str, str]]:
    table = block.find(
        "_pdbx_struct_assembly.",
        ["id", "details", "oligomeric_details"],
    )
    if not table:
        return {}

    meta: dict[str, tuple[str, str]] = {}
    for row in table:
        assembly_id = str(row[0]).strip()
        details = str(row[1]).strip() if not _is_null_token(str(row[1])) else ""
        oligo = str(row[2]).strip() if not _is_null_token(str(row[2])) else ""
        meta[assembly_id] = (details, oligo)

    return meta


def _extract_assembly_gens(block) -> dict[str, list[AssemblyGen]]:
    table = block.find(
        "_pdbx_struct_assembly_gen.",
        ["assembly_id", "oper_expression", "asym_id_list"],
    )
    if not table:
        return {}

    gens: dict[str, list[AssemblyGen]] = {}
    for row in table:
        assembly_id = str(row[0]).strip()
        oper_expr = str(row[1]).strip()
        asym_ids_raw = str(row[2]).strip()
        chain_ids = [c.strip() for c in asym_ids_raw.split(",") if c.strip()]
        gen = AssemblyGen(assembly_id=assembly_id, oper_expression=oper_expr, chain_ids=chain_ids)
        gens.setdefault(assembly_id, []).append(gen)

    return gens


def expand_oper_expression(expr: str) -> list[list[str]]:
    """
    oper_expression를 operation ID 목록의 리스트로 전개합니다.
    - 단순: "1,2,3"  → [["1"], ["2"], ["3"]]
    - 범위: "1-3"    → [["1"], ["2"], ["3"]]
    - 복합: "(1-2)(4,5)" → [["1","4"], ["1","5"], ["2","4"], ["2","5"]]
    """
    expr = expr.strip()
    groups = re.findall(r'\(([^)]+)\)', expr)
    if not groups:
        groups = [expr]

    parsed_groups: list[list[str]] = []
    for group in groups:
        ids: list[str] = []
        for token in group.split(","):
            token = token.strip()
            range_match = re.fullmatch(r'(\d+)-(\d+)', token)
            if range_match:
                start, end = int(range_match.group(1)), int(range_match.group(2))
                ids.extend(str(i) for i in range(start, end + 1))
            else:
                ids.append(token)
        parsed_groups.append(ids)

    if len(parsed_groups) == 1:
        return [[oid] for oid in parsed_groups[0]]

    # 데카르트 곱으로 전개
    result: list[list[str]] = [[]]
    for group in parsed_groups:
        result = [existing + [oid] for existing in result for oid in group]
    return result


def _get_block_value(block, tag: str) -> str | None:
    try:
        value = block.find_value(tag)
        if value and not _is_null_token(str(value)):
            return str(value).strip().strip("'\"")
    except Exception:
        pass
    return None


def _is_null_token(value: str) -> bool:
    return value.strip() in NULL_TOKENS


def count_input_stats(records: list[AtomRecord]) -> ConversionStats:
    residue_keys = {
        (record.model_num, record.chain_id, record.resseq_raw, record.ins_code, record.resname)
        for record in records
    }
    chain_keys = {(record.model_num, record.chain_id) for record in records}
    model_keys = {record.model_num for record in records}
    return ConversionStats(
        atoms=len(records),
        residues=len(residue_keys),
        chains=len(chain_keys),
        models=len(model_keys),
    )


def _is_null(value) -> bool:
    if value is None:
        return True
    return str(value).strip() in NULL_TOKENS


def _clean_token(value: str | None) -> str:
    if _is_null(value):
        return ""
    return str(value).strip()


def _get_value(atom_site, keys: Iterable[str], index: int, default: str = "") -> str:
    for key in keys:
        if key in atom_site:
            value = atom_site[key][index]
            if not _is_null(value):
                return str(value).strip()
    return default


def _parse_int(value: str | None) -> int | None:
    if _is_null(value):
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _parse_float(value: str | None, default: float = 0.0) -> float:
    if _is_null(value):
        return default
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default
