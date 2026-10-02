"""Inverse folding backed by the MPNN family (LigandMPNN repository).

This is an alternative to BoltzGen's built-in inverse folding model (BoltzIF).
It is selected with ``--inverse_fold_backend mpnn``; the default remains
``boltzif`` so existing behaviour is unchanged.

Why a pipeline step rather than a model swap
--------------------------------------------
BoltzGen's pipeline steps communicate purely through files. The inverse folding
step reads backbones from ``intermediate_designs/`` and writes sequence-carrying
structures to ``intermediate_designs_inverse_folded/``; every downstream step
(folding, analysis, filtering) reads that directory through the same
``FromGeneratedDataModule``. Replacing the step therefore only requires honouring
the on-disk contract, and avoids both the kNN-graph-shaped ``sample()``
signature of :class:`~boltzgen.model.modules.inverse_fold.InverseFoldingDecoder`
and the ``strict=True`` checkpoint loading in ``Boltz.load_from_checkpoint``.

The on-disk contract
--------------------
Input  : ``<design_dir>/<stem>.cif`` + ``<stem>.npz``
Output : ``<output_dir>/<stem>[_<idx>].cif`` + matching ``.npz``

The **sequence travels in the mmCIF residue names** (3-letter CCD codes), not in
the npz -- see ``DesignWriter`` in ``writer.py`` and ``analyze.py``, which
reconstructs sequences from the parsed structure. The npz carries only
token-indexed masks (``design_mask``, ``mol_type``, ``ss_type``,
``token_resolved_mask``, ``binding_type``, optionally
``inverse_fold_design_mask`` and ``aa_constraint_mask``). Inverse folding does
not change any of those masks, so this task copies the input npz forward
verbatim rather than recomputing it -- recomputation could only introduce
drift, and a token-count mismatch makes ``data_from_generated`` mark the sample
as an exception and silently drop it.

Residue addressing
------------------
LigandMPNN addresses residues as ``<chain><resnum>`` (e.g. ``A54``). Those
identifiers are read back from the **converted PDB**, never from the source
mmCIF: the converter may renumber a whole chain when any residue falls outside
the PDB-representable range, and stale identifiers would fix the wrong residues
-- silently letting the target sequence be redesigned.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
from omegaconf import OmegaConf

from boltzgen.data import const
from boltzgen.data.mol import load_canonicals
from boltzgen.data.parse.mmcif import parse_mmcif
from boltzgen.data.write.cif_to_pdb import convert_cif_to_pdb_detailed
from boltzgen.data.write.mmcif import to_mmcif
from boltzgen.task.predict.mpnn_alphabet import (
    MPNN_ALPHABET,
    MPNN_UNKNOWN_LETTER,
    canonical_index_to_mpnn_letter,
)
from boltzgen.task.task import Task

logger = logging.getLogger(__name__)

# Metadata keys copied verbatim from the input npz to the output npz.
# Required keys first; the optional ones are only present for some protocols.
_REQUIRED_METADATA_KEYS = (
    "design_mask",
    "mol_type",
    "ss_type",
    "token_resolved_mask",
    "binding_type",
)
_OPTIONAL_METADATA_KEYS = ("inverse_fold_design_mask", "aa_constraint_mask")

# Suffixes that mark a cif in the design dir as *not* being an input backbone.
# Mirrors the enumeration rules in data_from_generated.init_dataset.
_EXCLUDED_CIF_MARKERS = ("_native.cif", "_metadata.npz")


@dataclass
class ResidueKey:
    """A residue as LigandMPNN addresses it: chain + number + insertion code."""

    chain: str
    number: int
    icode: str = ""

    def as_mpnn(self) -> str:
        # LigandMPNN parses "A54"; insertion codes are appended directly.
        return f"{self.chain}{self.number}{self.icode}"


@dataclass
class DesignInputs:
    """One backbone to be inverse folded."""

    stem: str
    cif_path: Path
    npz_path: Path
    metadata: dict

    @property
    def design_mask(self) -> np.ndarray:
        """Token-level mask of positions the model may redesign."""
        if "inverse_fold_design_mask" in self.metadata:
            return self.metadata["inverse_fold_design_mask"].astype(bool)
        return self.metadata["design_mask"].astype(bool)


# Where the MPNN environment lives by default. LigandMPNN pins numpy==1.23.5
# and requires ProDy, while BoltzGen pins numpy==2.0.2, so the two cannot share
# an interpreter -- the MPNN backend always runs out-of-process. Both values are
# overridable from config or the environment, following the
# ``MDANALYSIS_PYTHON_PATH`` pattern in Curieus' bynum_converter.
#
# The defaults match the deployment layout: conda lives at
# /home/connects/miniforge3 (see the shared COMMON_ENV_SETUP, which sources
# etc/profile.d/conda.sh from there, and bioemu's HPACKER_PYTHONBIN), and model
# checkouts live under /home/connects/SCV_Models/Models (MODEL_DIR).
_DEFAULT_MPNN_PYTHON = "/home/connects/miniforge3/envs/ligandmpnn/bin/python"
_DEFAULT_MPNN_REPO = "/home/connects/SCV_Models/Models/LigandMPNN"

_MPNN_PYTHON_ENV_VAR = "LIGANDMPNN_PYTHONBIN"
_MPNN_REPO_ENV_VAR = "LIGANDMPNN_REPO_DIR"


@dataclass
class MPNNSettings:
    """Everything needed to translate BoltzGen constraints into MPNN flags."""

    model_type: str = "ligand_mpnn"
    checkpoint_path: str = ""
    weights_dir: str = ""
    temperature: float = 0.1
    num_sequences: int = 1
    seed: int | None = None
    # Globally forbidden amino acids (BoltzGen's --inverse_fold_avoid).
    omit_aa: str = ""
    verbose: int = 0
    extra_args: dict = field(default_factory=dict)
    # Out-of-process execution: interpreter and checkout of the MPNN repo.
    python_bin: str = ""
    repo_dir: str = ""

    def resolved_python_bin(self) -> str:
        """Interpreter for the MPNN environment (config > env var > default)."""
        return (
            self.python_bin
            or os.environ.get(_MPNN_PYTHON_ENV_VAR, "")
            or _DEFAULT_MPNN_PYTHON
        )

    def resolved_repo_dir(self) -> str:
        """Checkout of the LigandMPNN repository (config > env var > default)."""
        return (
            self.repo_dir
            or os.environ.get(_MPNN_REPO_ENV_VAR, "")
            or _DEFAULT_MPNN_REPO
        )

    def effective_omit_aa(self) -> str:
        """Forbidden residues, always including ``X``.

        ``X`` maps to ``UNK``, whose BoltzGen token id falls outside the
        canonical slice the sampler draws from, so it can never be written to a
        designed position. LigandMPNN also emits ``X`` for residues it fails to
        recognise, so excluding it here prevents that failure mode from reaching
        the output structure.
        """
        letters = {c for c in self.omit_aa.upper() if c in MPNN_ALPHABET}
        letters.add(MPNN_UNKNOWN_LETTER)
        return "".join(sorted(letters))


def discover_design_inputs(design_dir: Path) -> list[DesignInputs]:
    """Enumerate backbones in ``design_dir`` that still need inverse folding.

    Applies the same exclusion rules as ``FromGeneratedDataModule`` so the two
    backends see an identical set of inputs.
    """
    inputs: list[DesignInputs] = []
    for path in sorted(design_dir.iterdir()):
        if path.suffix not in (".cif", ".pdb"):
            continue
        if any(marker in path.name for marker in _EXCLUDED_CIF_MARKERS):
            continue

        npz_path = path.with_suffix(".npz")
        if not npz_path.exists():
            # Legacy layout written by older BoltzGen versions.
            legacy = path.with_name(f"{path.stem}_metadata.npz")
            if legacy.exists():
                npz_path = legacy
            else:
                logger.warning(
                    "skipping %s: no companion .npz metadata found", path.name
                )
                continue

        with np.load(npz_path) as handle:
            metadata = {key: handle[key] for key in handle.files}

        missing = [k for k in _REQUIRED_METADATA_KEYS if k not in metadata]
        if missing:
            logger.warning(
                "skipping %s: metadata is missing required keys %s",
                path.name,
                missing,
            )
            continue

        inputs.append(
            DesignInputs(
                stem=path.stem,
                cif_path=path,
                npz_path=npz_path,
                metadata=metadata,
            )
        )
    return inputs


def convert_cif_to_pdb_file(cif_path: Path, pdb_path: Path) -> list[str]:
    """Convert a design mmCIF to PDB, returning any conversion warnings.

    Uses ``compatible`` mode: ``strict`` would reject structures that the rest
    of the pipeline handles fine, while ``best_effort`` can wrap atom serials
    and produce duplicates.
    """
    result = convert_cif_to_pdb_detailed(
        cif_path.read_bytes(), cif_path.name, "compatible"
    )
    pdb_path.write_text(result.pdb_content)
    return list(result.warnings or [])


def read_pdb_residue_keys(pdb_path: Path) -> list[ResidueKey]:
    """Read residue identifiers from a PDB in file order, one per residue.

    Read from the converted PDB rather than the source mmCIF because the
    converter may renumber chains; see the module docstring.
    """
    keys: list[ResidueKey] = []
    seen: set[tuple[str, int, str]] = set()
    for line in pdb_path.read_text().splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        chain = line[21].strip()
        try:
            number = int(line[22:26])
        except ValueError:
            continue
        icode = line[26].strip()
        identity = (chain, number, icode)
        if identity in seen:
            continue
        seen.add(identity)
        keys.append(ResidueKey(chain=chain, number=number, icode=icode))
    return keys


def build_fixed_residue_spec(
    residue_keys: Sequence[ResidueKey], design_mask: np.ndarray
) -> str:
    """Build LigandMPNN's ``--fixed_residues`` value.

    Every position the design mask leaves False must be pinned to its input
    identity; BoltzGen's own decoder does this by seeding those positions from
    ``res_type_clone`` and excluding them from the decoding order.

    Raises:
        ValueError: if the mask and the structure disagree on residue count,
            which would misalign every identifier after the discrepancy.
    """
    if len(residue_keys) != len(design_mask):
        raise ValueError(
            f"Residue count mismatch: structure has {len(residue_keys)} residues "
            f"but design_mask has {len(design_mask)} entries. Refusing to build a "
            "fixed-residue spec that would address the wrong residues."
        )
    fixed = [
        key.as_mpnn()
        for key, designable in zip(residue_keys, design_mask)
        if not designable
    ]
    return " ".join(fixed)


def build_omit_aa_per_residue(
    residue_keys: Sequence[ResidueKey],
    design_mask: np.ndarray,
    aa_constraint_mask: np.ndarray | None,
) -> dict[str, str]:
    """Translate BoltzGen's ``(N, 20)`` constraint mask to MPNN per-residue omits.

    The constraint mask is expressed in BoltzGen's canonical index space, which
    is ordered alphabetically by 3-letter code -- a different order from MPNN's
    1-letter alphabet, so the mapping goes through an explicit table.

    A non-zero entry means "this amino acid is disallowed here". Positions that
    are not designable are skipped, since they are pinned anyway.
    """
    if aa_constraint_mask is None:
        return {}
    if aa_constraint_mask.ndim != 2 or aa_constraint_mask.shape[1] != 20:
        logger.warning(
            "ignoring aa_constraint_mask with unexpected shape %s (expected (N, 20))",
            getattr(aa_constraint_mask, "shape", "unknown"),
        )
        return {}
    if len(aa_constraint_mask) != len(residue_keys):
        logger.warning(
            "ignoring aa_constraint_mask: %d rows for %d residues",
            len(aa_constraint_mask),
            len(residue_keys),
        )
        return {}

    omit: dict[str, str] = {}
    for key, designable, row in zip(residue_keys, design_mask, aa_constraint_mask):
        if not designable:
            continue
        blocked = "".join(
            canonical_index_to_mpnn_letter(idx)
            for idx in range(20)
            if row[idx] > 0
        )
        if blocked:
            omit[key.as_mpnn()] = blocked
    return omit


def build_symmetry_spec(
    residue_keys: Sequence[ResidueKey],
    design_mask: np.ndarray,
    symmetric_group: np.ndarray | None,
    residue_index: np.ndarray | None,
) -> str:
    """Build LigandMPNN's ``--symmetry_residues`` value for homomer tying.

    Mirrors ``InverseFoldingDecoder._build_symmetric_groups``: designable
    positions sharing a (symmetric group, residue index) pair must receive the
    same amino acid. LigandMPNN expects groups as ``A1,B1|A2,B2``.
    """
    if symmetric_group is None or residue_index is None:
        return ""
    if not (len(symmetric_group) == len(residue_index) == len(residue_keys)):
        logger.warning("ignoring symmetry metadata: array lengths disagree")
        return ""

    groups: dict[tuple[int, int], list[str]] = {}
    for key, designable, group, res_idx in zip(
        residue_keys, design_mask, symmetric_group, residue_index
    ):
        if not designable:
            continue
        group_id = int(group)
        if group_id <= 0:  # 0 means "no symmetry group"
            continue
        groups.setdefault((group_id, int(res_idx)), []).append(key.as_mpnn())

    tied = [",".join(members) for members in groups.values() if len(members) > 1]
    return "|".join(tied)


def _three_letter_sequence(letters: Iterable[str]) -> list[str]:
    """Convert MPNN 1-letter codes to the 3-letter names used in mmCIF."""
    from boltzgen.task.predict.mpnn_alphabet import mpnn_letter_to_boltzgen_token

    return [mpnn_letter_to_boltzgen_token(letter) for letter in letters]


# run.py exposes one checkpoint flag per model type.
_CHECKPOINT_FLAGS = {
    "protein_mpnn": "--checkpoint_protein_mpnn",
    "ligand_mpnn": "--checkpoint_ligand_mpnn",
    "soluble_mpnn": "--checkpoint_soluble_mpnn",
    "global_label_membrane_mpnn": "--checkpoint_global_label_membrane_mpnn",
    "per_residue_label_membrane_mpnn": "--checkpoint_per_residue_label_membrane_mpnn",
}


def verify_mpnn_environment(settings: MPNNSettings) -> tuple[str, Path]:
    """Check the MPNN interpreter and repository exist before running anything.

    Failing here names the missing path, rather than surfacing an opaque
    subprocess error after a long run has already started.

    Returns:
        The interpreter path and the repository directory.

    Raises:
        FileNotFoundError: if either path is missing, or ``run.py`` is absent.
    """
    python_bin = settings.resolved_python_bin()
    repo_dir = Path(settings.resolved_repo_dir())

    if not Path(python_bin).exists():
        raise FileNotFoundError(
            f"MPNN interpreter not found: {python_bin}. Set it with "
            f"'--config inverse_folding mpnn.python_bin=<path>' or the "
            f"{_MPNN_PYTHON_ENV_VAR} environment variable. The MPNN backend "
            "needs its own conda environment because LigandMPNN pins "
            "numpy==1.23.5 and requires ProDy, which conflict with BoltzGen."
        )
    if not repo_dir.is_dir():
        raise FileNotFoundError(
            f"LigandMPNN repository not found: {repo_dir}. Set it with "
            f"'--config inverse_folding mpnn.repo_dir=<path>' or the "
            f"{_MPNN_REPO_ENV_VAR} environment variable."
        )
    run_script = repo_dir / "run.py"
    if not run_script.exists():
        raise FileNotFoundError(
            f"LigandMPNN run.py not found at {run_script}. "
            f"Is {repo_dir} really a LigandMPNN checkout?"
        )
    return python_bin, repo_dir


def build_mpnn_command(
    settings: MPNNSettings,
    repo_dir: Path,
    python_bin: str,
    pdb_multi_path: Path,
    out_folder: Path,
    fixed_residues_multi_path: Path | None = None,
    omit_per_residue_path: Path | None = None,
    symmetry_spec: str = "",
) -> list[str]:
    """Assemble the LigandMPNN command line.

    Uses the ``*_multi`` JSON inputs so a single invocation -- and therefore a
    single model load -- covers every design.
    """
    cmd = [
        python_bin,
        "-u",
        str(repo_dir / "run.py"),
        "--model_type",
        settings.model_type,
        "--pdb_path_multi",
        str(pdb_multi_path),
        "--out_folder",
        str(out_folder),
        "--temperature",
        str(settings.temperature),
        "--number_of_batches",
        str(settings.num_sequences),
        "--batch_size",
        "1",
        "--omit_AA",
        settings.effective_omit_aa(),
        "--verbose",
        str(settings.verbose),
        # Pin the output numbering instead of relying on run.py's default:
        # without this it starts at _1, and a later change to that default
        # would silently break output discovery again.
        "--zero_indexed",
        "1",
    ]
    if settings.seed is not None:
        cmd += ["--seed", str(settings.seed)]
    if fixed_residues_multi_path is not None:
        cmd += ["--fixed_residues_multi", str(fixed_residues_multi_path)]
    if omit_per_residue_path is not None:
        cmd += ["--omit_AA_per_residue_multi", str(omit_per_residue_path)]
    if symmetry_spec:
        cmd += ["--symmetry_residues", symmetry_spec]

    # Checkpoint flags default to './model_params/...' relative to the repo, so
    # pass an absolute path whenever the caller specified one.
    if settings.checkpoint_path:
        flag = _CHECKPOINT_FLAGS.get(settings.model_type)
        if flag:
            cmd += [flag, settings.checkpoint_path]
        else:
            logger.warning(
                "no checkpoint flag known for model_type=%s; using its default",
                settings.model_type,
            )

    for key, value in settings.extra_args.items():
        cmd += [f"--{key}", str(value)]
    return cmd


def run_mpnn(cmd: Sequence[str], repo_dir: Path) -> None:
    """Invoke LigandMPNN, surfacing its own diagnostics on failure.

    ``cwd`` is the repository root because ``run.py`` resolves its default
    checkpoint paths relative to the working directory.
    """
    logger.info("running MPNN: %s", " ".join(cmd))
    completed = subprocess.run(
        list(cmd),
        cwd=str(repo_dir),
        capture_output=True,
        text=True,
    )
    stdout = (completed.stdout or "").strip()
    if stdout:
        # LigandMPNN reports which residues it redesigned here; without it a
        # run that finishes suspiciously fast gives nothing to diagnose.
        logger.info("MPNN stdout:\n%s", stdout)

    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        hint = ""
        # A mis-provisioned environment is the most likely first-run failure;
        # name it explicitly instead of leaving a bare traceback.
        if "No module named" in stderr or "ImportError" in stderr:
            hint = (
                "\nThis usually means the MPNN conda environment is missing a "
                "dependency. LigandMPNN requires numpy==1.23.5, ProDy and "
                "torch; install them with 'pip3 install -r requirements.txt' "
                "inside that environment."
            )
        raise RuntimeError(
            f"LigandMPNN failed with exit code {completed.returncode}.\n"
            f"--- stderr ---\n{stderr}{hint}"
        )


def find_backbone_pdb(backbones: Path, stem: str, seq_idx: int) -> Path | None:
    """Locate the backbone PDB for one design/sequence pair.

    ``run.py`` numbers its output from 1 unless ``--zero_indexed`` is set, and
    we set it -- but accept either convention so a change to that flag's
    default cannot silently strand the outputs again.
    """
    for candidate in (
        backbones / f"{stem}_{seq_idx}.pdb",
        backbones / f"{stem}_{seq_idx + 1}.pdb",
    ):
        if candidate.exists():
            return candidate
    return None


def read_designed_sequence(backbone_pdb: Path) -> dict[tuple[str, int, str], str]:
    """Read designed residue names from an MPNN backbone PDB.

    Returns a mapping of ``(chain, resnum, icode) -> 3-letter residue name``.

    The backbone PDBs are preferred over the FASTA output: they carry chain
    identifiers and residue numbers explicitly, whereas the FASTA concatenates
    chains with a separator and would require reconstructing the chain order.
    """
    sequence: dict[tuple[str, int, str], str] = {}
    for line in backbone_pdb.read_text().splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        resname = line[17:20].strip()
        chain = line[21].strip()
        try:
            resnum = int(line[22:26])
        except ValueError:
            continue
        icode = line[26].strip()
        sequence.setdefault((chain, resnum, icode), resname)
    return sequence


def apply_sequence_to_structure(
    structure,
    design_mask: np.ndarray,
    designed_names: Sequence[str | None],
) -> int:
    """Write designed residue names into a parsed :class:`Structure`, in place.

    Rewriting the parsed structure -- rather than patching the mmCIF text --
    is what keeps the output readable. ``to_mmcif`` derives both
    ``_atom_site.label_comp_id`` and ``_entity_poly_seq.mon_id`` from the same
    ``residues["name"]`` field, and ``parse_polymer`` asserts those two agree.
    Editing only the atom table leaves the sequence table describing the
    pre-design residues, which is exactly the mismatch that assertion catches.

    Args:
        structure: a ``Structure`` from ``parse_mmcif(...).data``.
        design_mask: per-residue flags, aligned with ``structure.residues``.
        designed_names: 3-letter names for designable positions; ``None``
            wherever the residue must keep its current identity.

    Returns:
        The number of residues whose name changed.
    """
    residues = structure.residues
    if not (len(residues) == len(design_mask) == len(designed_names)):
        raise ValueError(
            f"Length mismatch: {len(residues)} residues, "
            f"{len(design_mask)} mask entries, {len(designed_names)} names. "
            "Refusing to rename residues that may not correspond."
        )

    changed = 0
    for idx, (designable, new_name) in enumerate(zip(design_mask, designed_names)):
        if not designable or not new_name:
            continue
        old_name = str(residues[idx]["name"])
        if old_name == new_name:
            continue
        if new_name not in const.token_ids:
            raise ValueError(
                f"Designed residue {new_name!r} at index {idx} is not a known "
                "BoltzGen token."
            )

        residues[idx]["name"] = new_name
        # The tokenizer reads res_type, not name, so both must move together.
        residues[idx]["res_type"] = const.token_ids[new_name]
        # atom_center is CA for every amino acid, but atom_disto is CA only for
        # glycine and CB otherwise -- so it has to follow a GLY<->non-GLY flip.
        residues[idx]["atom_disto"] = (
            residues[idx]["atom_idx"] + const.res_to_disto_atom_id[new_name]
        )
        changed += 1
    return changed


def rewrite_cif_with_sequence(
    cif_path: Path,
    residue_keys: Sequence[ResidueKey],
    design_mask: np.ndarray,
    designed: dict[tuple[str, int, str], str],
    mols: dict | None,
    moldir: str | None,
) -> tuple[str, int]:
    """Re-emit a design mmCIF carrying the MPNN-designed sequence.

    Parses with the same reader the downstream steps use, so a structure that
    cannot be read fails here rather than several steps later.

    Returns:
        The mmCIF text and the number of residues changed.

    Raises:
        ValueError: if a designable residue is absent from the MPNN output, or
            if the parsed residues cannot be aligned with ``residue_keys``.
    """
    designable_keys = {
        (key.chain, key.number, key.icode)
        for key, flag in zip(residue_keys, design_mask)
        if flag
    }
    missing = designable_keys - set(designed)
    if missing:
        sample = sorted(missing)[:5]
        raise ValueError(
            f"{len(missing)} designable residue(s) missing from the MPNN output, "
            f"e.g. {sample}. Refusing to write a structure with stale residues."
        )

    structure = parse_mmcif(
        str(cif_path), mols, moldir=moldir, use_original_res_idx=False
    ).data

    # residue_keys comes from the converted PDB and lists only present
    # residues, while the parsed structure also carries unresolved ones.
    present_indices = [
        i for i, res in enumerate(structure.residues) if res["is_present"]
    ]
    if len(present_indices) != len(residue_keys):
        raise ValueError(
            f"Cannot align designed sequence: the structure has "
            f"{len(present_indices)} present residue(s) but the converted PDB "
            f"yielded {len(residue_keys)}. Refusing to rename by position."
        )

    full_mask = np.zeros(len(structure.residues), dtype=bool)
    names: list[str | None] = [None] * len(structure.residues)
    for struct_idx, key, designable in zip(
        present_indices, residue_keys, design_mask
    ):
        if not designable:
            continue
        full_mask[struct_idx] = True
        names[struct_idx] = designed[(key.chain, key.number, key.icode)]

    changed = apply_sequence_to_structure(structure, full_mask, names)
    return to_mmcif(structure), changed


class MPNNInverseFold(Task):
    """Run inverse folding with an MPNN model instead of BoltzIF.

    ``main.py`` builds the task with ``hydra.utils.instantiate(config)``, which
    passes every top-level config key as a constructor argument, and then calls
    ``run(config)`` with the same config. The constructor therefore accepts the
    keys declared in ``mpnn_inverse_fold.yaml``; ``**kwargs`` absorbs anything
    an override adds so a stray key cannot break instantiation.
    """

    def __init__(
        self,
        data: Optional[dict] = None,
        mpnn: Optional[dict] = None,
        output: Optional[str] = None,
        **kwargs,
    ) -> None:
        self.data = data or {}
        self.mpnn = mpnn or {}
        self.output = output
        if kwargs:
            logger.debug("ignoring unused config keys: %s", sorted(kwargs))

    def run(self, config: OmegaConf) -> None:  # noqa: D102 - see module docstring
        design_dir_value = OmegaConf.select(config, "data.design_dir")
        if not design_dir_value:
            raise ValueError(
                "data.design_dir is not set; the MPNN inverse folding step needs "
                "the directory holding the generated backbones."
            )
        output_value = OmegaConf.select(config, "output")
        if not output_value:
            raise ValueError("output is not set for the MPNN inverse folding step.")

        design_dir = Path(design_dir_value)
        output_dir = Path(output_value)
        output_dir.mkdir(parents=True, exist_ok=True)

        settings = self._settings_from_config(config)
        skip_existing = bool(OmegaConf.select(config, "data.skip_existing") or False)

        # Re-emitting the design mmCIF goes through BoltzGen's own parser, which
        # needs the CCD components and any SMILES ligands the design step wrote
        # alongside the backbones.
        moldir = OmegaConf.select(config, "data.moldir") or OmegaConf.select(
            config, "data.cfg.moldir"
        )
        if not moldir:
            # Without it the parser reaches load_molecules(None, ...) and dies
            # on Path(None), several frames away from the actual cause.
            raise ValueError(
                "moldir is not set for the MPNN inverse folding step. Rewriting "
                "the design mmCIF parses it with BoltzGen's reader, which needs "
                "the CCD components; pass 'data.cfg.moldir=<path>'."
            )
        mols = load_canonicals(moldir)
        extra_mol_dir = design_dir / const.molecules_dirname
        if extra_mol_dir.is_dir():
            mols = dict(mols or {})
            for mol_path in extra_mol_dir.glob("*.pkl"):
                mols[mol_path.stem] = pickle.load(mol_path.open("rb"))

        inputs = discover_design_inputs(design_dir)
        if not inputs:
            raise RuntimeError(
                f"No designs with metadata found in {design_dir}. "
                "The design step must run before inverse folding."
            )

        logger.info(
            "MPNN inverse folding: %d designs, model_type=%s, sequences=%d",
            len(inputs),
            settings.model_type,
            settings.num_sequences,
        )
        python_bin, repo_dir = verify_mpnn_environment(settings)

        with tempfile.TemporaryDirectory(prefix="boltzgen_mpnn_") as tmp:
            workdir = Path(tmp)
            prepared = self._prepare_inputs(inputs, workdir)
            if not prepared:
                raise RuntimeError(
                    "No designs could be prepared for MPNN; see warnings above."
                )

            mpnn_out = workdir / "mpnn_out"
            cmd = build_mpnn_command(
                settings=settings,
                repo_dir=repo_dir,
                python_bin=python_bin,
                pdb_multi_path=self._write_pdb_multi(prepared, workdir),
                out_folder=mpnn_out,
                fixed_residues_multi_path=self._write_fixed_residues(prepared, workdir),
                omit_per_residue_path=self._write_omit_per_residue(prepared, workdir),
                symmetry_spec=self._combined_symmetry_spec(prepared),
            )
            run_mpnn(cmd, repo_dir)

            written = self._collect_outputs(
                prepared, mpnn_out, output_dir, settings, skip_existing, mols, moldir
            )

        logger.info("MPNN inverse folding wrote %d structures", written)

    # -- helpers ---------------------------------------------------------

    def _prepare_inputs(
        self, inputs: Sequence[DesignInputs], workdir: Path
    ) -> list[dict]:
        """Convert each design to PDB and translate its constraints.

        Returns one record per design that converted successfully. A design
        that fails conversion is skipped with a warning rather than aborting
        the batch, since one malformed structure should not lose the rest.
        """
        pdb_dir = workdir / "pdb_in"
        pdb_dir.mkdir(parents=True, exist_ok=True)

        prepared: list[dict] = []
        for item in inputs:
            pdb_path = pdb_dir / f"{item.stem}.pdb"
            try:
                warnings = convert_cif_to_pdb_file(item.cif_path, pdb_path)
            except Exception as exc:  # noqa: BLE001 - reported per design
                logger.warning("skipping %s: cif->pdb failed: %s", item.stem, exc)
                continue
            if warnings:
                # Chain-wide renumbering happens here; residue identifiers are
                # read back from the converted PDB so they stay consistent.
                logger.warning("%s: conversion warnings: %s", item.stem, warnings)

            residue_keys = read_pdb_residue_keys(pdb_path)
            design_mask = item.design_mask
            try:
                fixed = build_fixed_residue_spec(residue_keys, design_mask)
            except ValueError as exc:
                logger.warning("skipping %s: %s", item.stem, exc)
                continue

            prepared.append(
                {
                    "item": item,
                    "pdb_path": pdb_path,
                    "residue_keys": residue_keys,
                    "design_mask": design_mask,
                    "fixed": fixed,
                    "omit": build_omit_aa_per_residue(
                        residue_keys,
                        design_mask,
                        item.metadata.get("aa_constraint_mask"),
                    ),
                    "symmetry": build_symmetry_spec(
                        residue_keys,
                        design_mask,
                        item.metadata.get("symmetric_group"),
                        item.metadata.get("feature_residue_index"),
                    ),
                }
            )
        return prepared

    @staticmethod
    def _write_pdb_multi(prepared: Sequence[dict], workdir: Path) -> Path:
        """Write the JSON listing every input PDB (keys are what run.py reads)."""
        path = workdir / "pdb_multi.json"
        path.write_text(
            json.dumps({str(rec["pdb_path"]): "" for rec in prepared}, indent=2)
        )
        return path

    @staticmethod
    def _write_fixed_residues(prepared: Sequence[dict], workdir: Path) -> Path | None:
        mapping = {
            str(rec["pdb_path"]): rec["fixed"] for rec in prepared if rec["fixed"]
        }
        if not mapping:
            return None
        path = workdir / "fixed_residues.json"
        path.write_text(json.dumps(mapping, indent=2))
        return path

    @staticmethod
    def _write_omit_per_residue(prepared: Sequence[dict], workdir: Path) -> Path | None:
        mapping = {
            str(rec["pdb_path"]): rec["omit"] for rec in prepared if rec["omit"]
        }
        if not mapping:
            return None
        path = workdir / "omit_aa_per_residue.json"
        path.write_text(json.dumps(mapping, indent=2))
        return path

    @staticmethod
    def _combined_symmetry_spec(prepared: Sequence[dict]) -> str:
        """Combine per-design symmetry groups.

        ``--symmetry_residues`` has no per-PDB JSON variant, so it can only be
        applied when a single design needs it. Applying one design's groups to
        the whole batch would tie unrelated positions together.
        """
        with_symmetry = [rec for rec in prepared if rec["symmetry"]]
        if not with_symmetry:
            return ""
        if len(prepared) == 1:
            return with_symmetry[0]["symmetry"]
        logger.warning(
            "symmetry tying requested for %d design(s) but --symmetry_residues "
            "applies to the whole batch; skipping it to avoid tying unrelated "
            "positions. Run those designs individually to apply symmetry.",
            len(with_symmetry),
        )
        return ""

    def _collect_outputs(
        self,
        prepared: Sequence[dict],
        mpnn_out: Path,
        output_dir: Path,
        settings: MPNNSettings,
        skip_existing: bool,
        mols: dict | None = None,
        moldir: str | None = None,
    ) -> int:
        """Write the sequence-carrying cif/npz pairs the pipeline expects."""
        backbones = mpnn_out / "backbones"
        if not backbones.is_dir():
            raise RuntimeError(
                f"MPNN produced no backbones directory at {backbones}. "
                "The run may have failed silently."
            )

        total = len(prepared) * settings.num_sequences
        num_digits = len(str(max(total - 1, 0))) if total > 1 else 0
        written = 0

        for design_idx, rec in enumerate(prepared):
            item: DesignInputs = rec["item"]
            for seq_idx in range(settings.num_sequences):
                backbone_pdb = find_backbone_pdb(backbones, item.stem, seq_idx)
                if backbone_pdb is None:
                    logger.warning(
                        "missing MPNN output for %s sequence %d in %s; skipping",
                        item.stem,
                        seq_idx,
                        backbones,
                    )
                    continue

                global_idx = design_idx * settings.num_sequences + seq_idx
                if total > 1:
                    name = f"{item.stem}_{global_idx:0{num_digits}d}"
                else:
                    name = item.stem

                out_cif = output_dir / f"{name}.cif"
                out_npz = output_dir / f"{name}.npz"
                if skip_existing and out_cif.exists() and out_npz.exists():
                    continue

                designed = read_designed_sequence(backbone_pdb)
                new_cif, changed = rewrite_cif_with_sequence(
                    item.cif_path,
                    rec["residue_keys"],
                    rec["design_mask"],
                    designed,
                    mols,
                    moldir,
                )
                out_cif.write_text(new_cif)
                # The masks are unchanged by inverse folding, so copy them
                # rather than recomputing and risking drift.
                shutil.copyfile(item.npz_path, out_npz)
                logger.debug("%s: %d residues redesigned", name, changed)
                written += 1

        if written == 0:
            # Returning 0 quietly would let the pipeline continue on the
            # pre-design backbones, and the failure would only surface much
            # later as a parse error in the folding step.
            produced = sorted(p.name for p in backbones.glob("*.pdb"))[:10]
            raise RuntimeError(
                f"MPNN produced no usable designs: expected "
                f"{len(prepared)} design(s) x {settings.num_sequences} sequence(s) "
                f"under {backbones}, but matched none. "
                f"Files present: {produced or 'none'}."
            )
        return written

    @staticmethod
    def _settings_from_config(config: OmegaConf) -> MPNNSettings:
        mpnn_cfg = OmegaConf.select(config, "mpnn") or {}
        return MPNNSettings(
            model_type=str(mpnn_cfg.get("model_type", "ligand_mpnn")),
            checkpoint_path=str(mpnn_cfg.get("checkpoint_path", "") or ""),
            weights_dir=str(mpnn_cfg.get("weights_dir", "") or ""),
            temperature=float(mpnn_cfg.get("temperature", 0.1)),
            num_sequences=int(mpnn_cfg.get("num_sequences", 1)),
            seed=mpnn_cfg.get("seed"),
            omit_aa=str(mpnn_cfg.get("omit_aa", "") or ""),
            verbose=int(mpnn_cfg.get("verbose", 0)),
            # Empty values fall through to the environment variables and then
            # the deployment defaults; see resolved_python_bin/resolved_repo_dir.
            python_bin=str(mpnn_cfg.get("python_bin", "") or ""),
            repo_dir=str(mpnn_cfg.get("repo_dir", "") or ""),
        )
