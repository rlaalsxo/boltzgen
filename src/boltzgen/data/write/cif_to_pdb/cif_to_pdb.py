from __future__ import annotations

import logging
import os
import tempfile

from .cif_to_pdb_parser import (
    count_input_stats,
    extract_assembly_info,
    extract_atom_records,
    extract_cryst_info,
    normalize_mode,
    read_gemmi_block,
)
from .cif_to_pdb_policy import apply_policy
from .cif_to_pdb_types import CifToPdbConversionResult, ConversionMode
from .cif_to_pdb_writer import count_output_stats, render_pdb

logger = logging.getLogger(__name__)


def convert_cif_to_pdb(
    cif_content: bytes,
    original_filename: str,
    mode: ConversionMode = "compatible",
) -> str:
    return convert_cif_to_pdb_detailed(cif_content, original_filename, mode).pdb_content


def convert_cif_to_pdb_detailed(
    cif_content: bytes,
    original_filename: str,
    mode: ConversionMode = "compatible",
) -> CifToPdbConversionResult:
    _validate_filename(original_filename)
    normalized_mode = normalize_mode(mode)

    gemmi_error: Exception | None = None
    try:
        return _convert_with_gemmi(cif_content, normalized_mode)
    except ImportError as exc:
        gemmi_error = exc
    except Exception as exc:
        gemmi_error = exc

    fallback_warning = "Gemmi conversion unavailable; used BioPython fallback."
    if gemmi_error:
        logger.warning(
            "cif_to_pdb_gemmi_failed: error_type=%s error_message=%s",
            type(gemmi_error).__name__,
            str(gemmi_error),
            exc_info=gemmi_error,
        )
        fallback_warning = f"{fallback_warning} ({type(gemmi_error).__name__}: {gemmi_error})"
    return _convert_with_biopython(cif_content, original_filename, normalized_mode, fallback_warning)


def _validate_filename(original_filename: str):
    lower_name = original_filename.lower()
    if not (lower_name.endswith(".cif") or lower_name.endswith(".mmcif")):
        raise ValueError(f"Unsupported file extension: {original_filename}")


def _convert_with_gemmi(cif_content: bytes, mode: ConversionMode) -> CifToPdbConversionResult:
    warnings: list[str] = []
    block = read_gemmi_block(cif_content)
    atom_site = block.get_mmcif_category("_atom_site.")
    records = extract_atom_records(atom_site)
    if not records:
        raise ValueError("The CIF file does not contain ATOM/HETATM records.")

    cryst_info = extract_cryst_info(block)
    assemblies = extract_assembly_info(block)

    formatted_records = apply_policy(records, mode, warnings)
    pdb_content = render_pdb(formatted_records, cryst_info=cryst_info, assemblies=assemblies or None)
    output_stats = count_output_stats(formatted_records)
    if output_stats.atoms == 0:
        raise ValueError("The converted PDB content is empty.")

    return CifToPdbConversionResult(
        pdb_content=pdb_content,
        mode=mode,
        engine="gemmi",
        warnings=warnings,
        input_stats=count_input_stats(records),
        output_stats=output_stats,
    )


def _convert_with_biopython(
    cif_content: bytes,
    original_filename: str,
    mode: ConversionMode,
    warning: str,
) -> CifToPdbConversionResult:
    from Bio.PDB import MMCIFParser, PDBIO

    lower_name = original_filename.lower()
    suffix = ".mmcif" if lower_name.endswith(".mmcif") else ".cif"
    tmp_in = None
    tmp_out = None

    try:
        tmp_in = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp_in.write(cif_content)
        tmp_in.close()

        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure("struct", tmp_in.name)

        tmp_out = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False)
        tmp_out.close()

        io = PDBIO()
        io.set_structure(structure)
        io.save(tmp_out.name)

        with open(tmp_out.name, "r", encoding="utf-8") as handle:
            pdb_content = handle.read()

        if not pdb_content or len(pdb_content.strip()) < 10:
            raise ValueError("The converted PDB content is empty.")

        return CifToPdbConversionResult(
            pdb_content=pdb_content,
            mode=mode,
            engine="biopython",
            warnings=[warning, "Mode-specific policy checks were skipped in fallback mode."],
        )
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"CIF to PDB conversion failed: {exc}") from exc
    finally:
        if tmp_in and os.path.exists(tmp_in.name):
            os.unlink(tmp_in.name)
        if tmp_out and os.path.exists(tmp_out.name):
            os.unlink(tmp_out.name)
