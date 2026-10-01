"""mmCIF -> PDB conversion with explicit PDB-format-limit policy.

LigandMPNN reads PDB files, while BoltzGen writes mmCIF, so the MPNN
inverse-folding backend needs a conversion step that preserves residue
identity. Chain IDs and residue numbers must survive the round trip exactly:
``--fixed_residues`` addresses residues as ``<chain><resnum>``, so any
renaming or renumbering would silently fix the wrong residues and let the
target sequence be redesigned.

This package is a **vendored copy** of ``server/services/cif_to_pdb*.py`` from
the Curieus repository. It is copied rather than imported because BoltzGen
runs standalone and cannot depend on the Curieus server package.

Only one adaptation was made to the copied sources: the server's structured
logger (``core.logging_config.get_logger``) was replaced with the standard
library ``logging`` module.

Why this implementation rather than a plain ``gemmi.write_pdb`` call:

- It uses gemmi purely as a *parser* (``_atom_site`` category) and formats PDB
  lines itself, so ``setup_entities()`` / ``write_pdb()`` never get a chance to
  rename chains or split subchains.
- It prefers ``auth_asym_id`` / ``auth_seq_id`` over their ``label_`` variants,
  keeping author chain names and original residue numbering.
- It preserves insertion codes and includes them in residue identity, which
  matters for antibody CDR numbering (100A/100B/100C).
- It checks PDB format limits explicitly (>62 chains, residue numbers outside
  [-999, 9999], >99999 atoms) and reports them through ``mode`` rather than
  letting the data be silently truncated.

Note the ``compatible``/``best_effort`` renumbering rule: if *any* residue in a
chain falls outside the representable range, that **entire chain** is
renumbered from 1 (``cif_to_pdb_policy.py``). Callers that need to address
residues afterwards should therefore derive their residue identifiers from the
converted PDB, not from the source mmCIF, and should inspect
``CifToPdbConversionResult.warnings``.
"""

from .cif_to_pdb import convert_cif_to_pdb, convert_cif_to_pdb_detailed
from .cif_to_pdb_types import CifToPdbConversionResult, ConversionMode

__all__ = [
    "CifToPdbConversionResult",
    "ConversionMode",
    "convert_cif_to_pdb",
    "convert_cif_to_pdb_detailed",
]
