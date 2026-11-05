# Added in this fork on 2026-11-05: tests for the pycdxml-smiles-to-cdxml command-line tool.
import json
import subprocess
import sys

import pytest
from rdkit import Chem

from pycdxml.cdxml_converter import mol_to_document


def _run(stdin, *args):
    return subprocess.run([sys.executable, "-m", "pycdxml.smiles_to_cdxml", *args], input=stdin,
                          capture_output=True, text=True)


def test_options_are_passed_to_parser_and_converter():
    smiles = "C[C@H](N)C(=O)O"
    result = _run(json.dumps([smiles]), "--no-remove-hs", "--add-chiral-hs", "--no-include-enhanced-stereo",
                  "--no-crossed-bonds")
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)

    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles(smiles, params)
    expected = mol_to_document(mol, add_chiral_hs=True, include_enhanced_stereo=False,
                               crossed_bonds=False).to_cdxml()
    assert output == [expected]


def test_failed_items_are_null_and_reported_without_smiles():
    result = _run(json.dumps(["CCO", "not a smiles", "", "C1CC1\r", "c1ccccc1"]))
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    assert len(output) == 5
    assert output[1] is None
    assert output[2] is None
    for index in (0, 3, 4):
        assert "<CDXML" in output[index]
    assert "item 1:" in result.stderr
    assert "item 2:" in result.stderr
    assert "not a smiles" not in result.stderr


@pytest.mark.parametrize("stdin", ["CCO\n", '{"smiles": "CCO"}', '["CCO", 1]'])
def test_invalid_input_fails(stdin):
    result = _run(stdin)
    assert result.returncode != 0
    assert result.stdout == ""
    assert result.stderr


def test_help_shows_defaults():
    result = _run("", "--help")
    assert result.returncode == 0
    assert "--no-remove-hs" in result.stdout
    assert "(default: off)" in result.stdout


@pytest.mark.parametrize("smiles, option", [("[H]OC", "--no-remove-hs"),
                                            ("[C@@H]12CC[C@@H](CC1)C2", "--add-chiral-hs"),
                                            ("c1ccccc1", "--no-kekulize"),
                                            ("C[C@H](N)C(=O)O", "--no-add-wedge-bonds"),
                                            ("C[C@H](N)C(=O)O |&1:1|", "--no-include-enhanced-stereo"),
                                            ("N", "--no-add-hydrogens")])
def test_option_changes_output(smiles, option):
    default = _run(json.dumps([smiles]))
    changed = _run(json.dumps([smiles]), option)
    assert default.returncode == 0 and changed.returncode == 0
    assert json.loads(default.stdout) != json.loads(changed.stdout)
