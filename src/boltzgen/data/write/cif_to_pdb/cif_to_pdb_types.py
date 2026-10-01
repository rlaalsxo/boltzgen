from __future__ import annotations

import string
from dataclasses import dataclass, field
from typing import Literal


ConversionMode = Literal["strict", "compatible", "best_effort"]

SUPPORTED_MODES = {"strict", "compatible", "best_effort"}
NULL_TOKENS = {"", ".", "?"}
MAX_ATOM_SERIAL = 99999
MAX_RESSEQ = 9999
MIN_RESSEQ = -999
PDB_CHAIN_IDS = string.ascii_uppercase + string.ascii_lowercase + string.digits


@dataclass
class ConversionStats:
    atoms: int = 0
    residues: int = 0
    chains: int = 0
    models: int = 0


@dataclass
class AtomRecord:
    source_index: int
    record_type: str
    original_serial: int | None
    atom_name: str
    altloc: str
    resname: str
    chain_id: str
    resseq_raw: str
    ins_code: str
    x: float
    y: float
    z: float
    occupancy: float
    bfactor: float
    element: str
    charge: str
    model_num: int


@dataclass
class FormattedAtomRecord:
    source_index: int
    record_type: str
    serial: int
    atom_name: str
    altloc: str
    resname: str
    chain_id: str
    resseq: int
    ins_code: str
    x: float
    y: float
    z: float
    occupancy: float
    bfactor: float
    element: str
    charge: str


@dataclass
class CrystInfo:
    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float
    space_group: str
    z_value: int = 1


@dataclass
class AssemblyOperation:
    oper_id: str
    matrix: list[list[float]]  # 3x3
    vector: list[float]         # 3


@dataclass
class AssemblyGen:
    assembly_id: str
    oper_expression: str
    chain_ids: list[str]


@dataclass
class AssemblyInfo:
    assembly_id: str
    details: str
    oligomeric_details: str
    gens: list[AssemblyGen]
    operations: dict[str, AssemblyOperation]


@dataclass
class CifToPdbConversionResult:
    pdb_content: str
    mode: str
    engine: str
    warnings: list[str] = field(default_factory=list)
    input_stats: ConversionStats = field(default_factory=ConversionStats)
    output_stats: ConversionStats = field(default_factory=ConversionStats)
