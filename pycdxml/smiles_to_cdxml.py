# Added in this fork on 2026-11-05: command-line tool that converts SMILES to cdxml.
"""
Convert SMILES strings to cdxml documents.

Input: one JSON list of SMILES strings on stdin.

Output: one JSON list on stdout, with one entry for each input item, in input order. The entry is the cdxml string
of the item, or null if the item failed. For each failed item, one line goes to stderr:
``item <index>: <exception type>: <message>``. These lines do not contain the SMILES, and RDKit's own log messages
are turned off.

Exit code: 0 if the options and the input are valid, even if some items fail. If stdin is not a JSON list of
strings, a message goes to stderr, nothing goes to stdout and the exit code is not 0.

Example::

    echo '["CCO", "c1ccccc1"]' | pycdxml-smiles-to-cdxml --no-kekulize
"""
import argparse
import json
import sys

from rdkit import Chem
from rdkit import RDLogger

from pycdxml.cdxml_converter import mol_to_document

# (name, default, help text)
_SWITCHES = [
    ("remove-hs", True, "let the SMILES parser remove explicit hydrogens"),
    ("kekulize", True, "kekulize the molecule before drawing"),
    ("add-chiral-hs", False, "add explicit hydrogens at chiral centres before drawing"),
    ("add-wedge-bonds", True, "set wedge and hash bonds for stereo centres before drawing"),
    ("include-enhanced-stereo", True, "include enhanced stereo in the cdxml"),
    ("crossed-bonds", True, "draw double bonds with undefined stereo as crossed bonds"),
    ("add-hydrogens", True, "prepare the molecule for drawing and show hydrogen counts in atom labels"),
]


class _InputError(Exception):
    pass


class _SwitchAction(argparse.Action):
    """Sets the destination to True for --<name> and to False for --no-<name>."""

    def __init__(self, option_strings, dest, default=None, help=None):
        super().__init__(option_strings, dest, nargs=0, default=default, help=help)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, not option_string.startswith("--no-"))


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        prog="pycdxml-smiles-to-cdxml",
        description="Read a JSON list of SMILES strings from stdin and write a JSON list of cdxml strings "
                    "(null for each item that failed) to stdout.",
    )
    for name, default, help_text in _SWITCHES:
        parser.add_argument(f"--{name}", f"--no-{name}", action=_SwitchAction, default=default,
                            help=f"{help_text} (default: {'on' if default else 'off'})")
    return parser.parse_args(argv)


def _read_smiles_list(text):
    try:
        value = json.loads(text)
    except json.JSONDecodeError as e:
        raise _InputError(f"stdin is not valid JSON: {e}") from None
    if not isinstance(value, list):
        raise _InputError(f"stdin must be a JSON list of strings, not a JSON {type(value).__name__}")
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise _InputError(f"stdin must be a JSON list of strings, but item {index} is a {type(item).__name__}")
    return value


def _convert(smiles, args):
    if not smiles:
        raise ValueError("empty SMILES")
    params = Chem.SmilesParserParams()
    params.removeHs = args.remove_hs
    mol = Chem.MolFromSmiles(smiles, params)
    if mol is None:
        raise ValueError("RDKit could not parse the SMILES")
    doc = mol_to_document(mol,
                          include_enhanced_stereo=args.include_enhanced_stereo,
                          crossed_bonds=args.crossed_bonds,
                          kekulize_molecule=args.kekulize,
                          add_chiral_hs=args.add_chiral_hs,
                          add_wedge_bonds=args.add_wedge_bonds,
                          add_hydrogens=args.add_hydrogens)
    return doc.to_cdxml()


def main(argv=None):
    args = _parse_args(argv)
    # RDKit log messages can contain the SMILES
    RDLogger.DisableLog("rdApp.*")

    try:
        smiles_list = _read_smiles_list(sys.stdin.read())
    except _InputError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    results = []
    for index, smiles in enumerate(smiles_list):
        try:
            results.append(_convert(smiles, args))
        except Exception as e:
            print(f"item {index}: {type(e).__name__}: {e}", file=sys.stderr)
            results.append(None)

    json.dump(results, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
