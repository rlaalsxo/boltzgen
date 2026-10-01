"""Alphabet translation between BoltzGen tokens and the MPNN (LigandMPNN) alphabet.

BoltzGen and LigandMPNN both represent the 20 canonical amino acids, but in
different orders and different widths:

- BoltzGen uses a 33-entry vocabulary (``const.tokens``) of 3-letter CCD codes.
  The canonical amino acids occupy the contiguous slice
  ``[canonicals_offset : canonicals_offset + 20]`` and are ordered
  alphabetically by 3-letter code (ALA, ARG, ASN, ASP, CYS, ...).
- LigandMPNN uses a 21-entry alphabet of 1-letter codes ordered alphabetically
  by *1-letter* code (A, C, D, E, F, ...), with ``X`` last.

Because the two orderings differ, index arithmetic cannot be used to convert
between them; an explicit permutation table is required.

``X``/``UNK`` deserves special care: it is representable in BoltzGen's full
33-token vocabulary (as ``UNK``) but falls *outside* the canonical slice that
the inverse-folding sampler draws from. A designed position must therefore
never be assigned ``X`` -- see :data:`MPNN_UNKNOWN_LETTER` and the ``omit_AA``
wiring in the MPNN inverse-folding task.
"""

from boltzgen.data import const

# LigandMPNN's alphabet, mirroring ``restype_str_to_int`` in the upstream
# repository's ``data_utils.py``. Index == model output index.
MPNN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"

# The sentinel LigandMPNN emits for residues it cannot identify. It must be
# excluded from design output because it has no canonical BoltzGen counterpart.
MPNN_UNKNOWN_LETTER = "X"


def mpnn_letter_to_boltzgen_token(letter: str) -> str:
    """Map a LigandMPNN 1-letter code to a BoltzGen 3-letter token.

    Raises:
        KeyError: if the letter is not part of the MPNN alphabet.
    """
    if letter not in MPNN_ALPHABET:
        raise KeyError(f"Not an MPNN alphabet letter: {letter!r}")
    return const.prot_letter_to_token[letter]


def mpnn_letter_to_canonical_index(letter: str) -> int:
    """Map a LigandMPNN 1-letter code to BoltzGen's 20-wide canonical index.

    This is the index space used by ``aa_constraint_mask`` and by
    ``inverse_fold_restriction``.

    Raises:
        ValueError: for ``X``, which has no canonical counterpart.
    """
    token = mpnn_letter_to_boltzgen_token(letter)
    if token not in const.canonical_tokens:
        raise ValueError(
            f"MPNN letter {letter!r} maps to {token!r}, which is not a canonical "
            "amino acid and cannot be represented in the canonical index space."
        )
    return const.canonical_tokens.index(token)


def mpnn_letter_to_token_id(letter: str) -> int:
    """Map a LigandMPNN 1-letter code to BoltzGen's 33-wide global token id."""
    return const.token_ids[mpnn_letter_to_boltzgen_token(letter)]


def boltzgen_token_to_mpnn_letter(token: str) -> str:
    """Map a BoltzGen 3-letter token to a LigandMPNN 1-letter code.

    Non-protein tokens (nucleotides) and unknown protein tokens collapse to
    ``X``, matching LigandMPNN's own handling of unrecognised residues.
    """
    return const.prot_token_to_letter.get(token, MPNN_UNKNOWN_LETTER)


def canonical_index_to_mpnn_letter(index: int) -> str:
    """Map a BoltzGen canonical (20-wide) index to a LigandMPNN 1-letter code."""
    if not 0 <= index < len(const.canonical_tokens):
        raise IndexError(
            f"Canonical index {index} out of range "
            f"[0, {len(const.canonical_tokens)})."
        )
    return boltzgen_token_to_mpnn_letter(const.canonical_tokens[index])


def build_canonical_to_mpnn_permutation() -> list[int]:
    """Permutation from BoltzGen canonical index -> MPNN alphabet index.

    ``perm[i]`` is the MPNN index of the amino acid that BoltzGen stores at
    canonical index ``i``. Useful for reordering a whole ``(N, 20)`` constraint
    matrix in one vectorised step rather than per residue.
    """
    return [
        MPNN_ALPHABET.index(canonical_index_to_mpnn_letter(i))
        for i in range(len(const.canonical_tokens))
    ]


def build_mpnn_to_canonical_permutation() -> list[int]:
    """Permutation from MPNN alphabet index -> BoltzGen canonical index.

    ``X`` has no canonical counterpart and is reported as ``-1`` so callers can
    detect and reject it explicitly instead of silently folding it into a real
    amino acid.
    """
    result = []
    for letter in MPNN_ALPHABET:
        if letter == MPNN_UNKNOWN_LETTER:
            result.append(-1)
        else:
            result.append(mpnn_letter_to_canonical_index(letter))
    return result
