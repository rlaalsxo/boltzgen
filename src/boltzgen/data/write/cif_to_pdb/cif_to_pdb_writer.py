from __future__ import annotations

from .cif_to_pdb_types import AssemblyInfo, ConversionStats, CrystInfo, FormattedAtomRecord
from .cif_to_pdb_parser import expand_oper_expression


def count_output_stats(records: list[FormattedAtomRecord]) -> ConversionStats:
    residue_keys = {
        (record.chain_id, record.resseq, record.ins_code, record.resname)
        for record in records
    }
    chain_keys = {record.chain_id for record in records}
    return ConversionStats(
        atoms=len(records),
        residues=len(residue_keys),
        chains=len(chain_keys),
        models=1,
    )


def render_pdb(
    records: list[FormattedAtomRecord],
    cryst_info: CrystInfo | None = None,
    assemblies: list[AssemblyInfo] | None = None,
) -> str:
    lines: list[str] = []

    if cryst_info is not None:
        lines.append(_format_cryst1(cryst_info))

    if assemblies:
        for assembly in assemblies:
            lines.extend(_format_remark_350(assembly))

    current_chain: str | None = None
    for record in records:
        if current_chain is not None and record.chain_id != current_chain:
            lines.append("TER\n")
        current_chain = record.chain_id
        lines.append(_format_atom_line(record))

    lines.append("TER\n")
    lines.append("END\n")
    return "".join(lines)


def _format_cryst1(info: CrystInfo) -> str:
    return (
        f"CRYST1"
        f"{info.a:>9.3f}"
        f"{info.b:>9.3f}"
        f"{info.c:>9.3f}"
        f"{info.alpha:>7.2f}"
        f"{info.beta:>7.2f}"
        f"{info.gamma:>7.2f} "
        f"{info.space_group:<11}"
        f"{info.z_value:>4}"
        f"\n"
    )


def _format_remark_350(assembly: AssemblyInfo) -> list[str]:
    lines: list[str] = []
    aid = assembly.assembly_id

    lines.append(f"REMARK 350 BIOMOLECULE: {aid}\n")
    if assembly.details:
        lines.append(f"REMARK 350 AUTHOR DETERMINED BIOLOGICAL UNIT: {assembly.details.upper()}\n")
    if assembly.oligomeric_details:
        lines.append(f"REMARK 350 SOFTWARE DETERMINED QUATERNARY STRUCTURE: {assembly.oligomeric_details.upper()}\n")

    # 각 gen마다 체인 목록 + BIOMT 행렬 출력
    biomt_serial = 1
    for gen in assembly.gens:
        # asym_id → auth chain_id 매핑은 현재 범위 밖이므로 asym_id 그대로 사용
        chain_str = ", ".join(gen.chain_ids)
        lines.append(f"REMARK 350 APPLY THE FOLLOWING TO CHAINS: {chain_str}\n")

        oper_sets = expand_oper_expression(gen.oper_expression)
        for oper_ids in oper_sets:
            # 복합 expression이면 행렬 곱으로 합성
            combined_matrix, combined_vector = _compose_operations(oper_ids, assembly.operations)
            if combined_matrix is None:
                continue

            for row_idx in range(3):
                m = combined_matrix[row_idx]
                v = combined_vector[row_idx]
                lines.append(
                    f"REMARK 350   BIOMT{row_idx + 1}"
                    f"{biomt_serial:>4}"
                    f"  {m[0]:>10.6f}{m[1]:>10.6f}{m[2]:>10.6f}"
                    f"     {v:>10.5f}\n"
                )
            biomt_serial += 1

    return lines


def _compose_operations(
    oper_ids: list[str],
    operations: dict,
) -> tuple[list[list[float]] | None, list[float] | None]:
    """여러 operation을 순서대로 행렬 곱으로 합성합니다."""
    # 항등 행렬 / 영 벡터에서 시작
    mat = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    vec = [0.0, 0.0, 0.0]

    for oid in oper_ids:
        op = operations.get(oid)
        if op is None:
            return None, None
        mat, vec = _matmul_op(op.matrix, op.vector, mat, vec)

    return mat, vec


def _matmul_op(
    m1: list[list[float]], v1: list[float],
    m2: list[list[float]], v2: list[float],
) -> tuple[list[list[float]], list[float]]:
    """(m1, v1) ∘ (m2, v2) = (m1·m2, m1·v2 + v1)"""
    result_m = [[0.0] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            result_m[i][j] = sum(m1[i][k] * m2[k][j] for k in range(3))

    result_v = [
        sum(m1[i][k] * v2[k] for k in range(3)) + v1[i]
        for i in range(3)
    ]
    return result_m, result_v


def _format_atom_line(record: FormattedAtomRecord) -> str:
    atom_name = _format_atom_name(record.atom_name, record.element)
    altloc = record.altloc[:1] if record.altloc else " "
    chain_id = record.chain_id[:1] if record.chain_id else " "
    ins_code = record.ins_code[:1] if record.ins_code else " "
    charge = record.charge.rjust(2) if record.charge else "  "

    return (
        f"{record.record_type:<6}"
        f"{record.serial:>5} "
        f"{atom_name}"
        f"{altloc}"
        f"{record.resname:>3} "
        f"{chain_id}"
        f"{record.resseq:>4}"
        f"{ins_code}   "
        f"{record.x:>8.3f}"
        f"{record.y:>8.3f}"
        f"{record.z:>8.3f}"
        f"{record.occupancy:>6.2f}"
        f"{record.bfactor:>6.2f}"
        f"          "
        f"{record.element:>2}"
        f"{charge}\n"
    )


def _format_atom_name(atom_name: str, element: str) -> str:
    atom_name = atom_name[:4]
    if len(atom_name) == 4:
        return atom_name
    if atom_name and atom_name[0].isdigit():
        return atom_name.rjust(4)
    if len(element.strip()) == 1:
        return f" {atom_name:<3}"
    return atom_name.ljust(4)
