# Added in this fork on 2026-11-05: tests for the drawing options of mol_to_document.
import pytest
from lxml import etree as ET
from rdkit import Chem

from pycdxml.cdxml_converter import mol_to_document


def _cdxml(smiles, **kwargs):
    return mol_to_document(Chem.MolFromSmiles(smiles), **kwargs).to_cdxml()


def _labels(cdxml):
    root = ET.fromstring(cdxml.encode("utf-8"))
    return [s.text for s in root.iterfind(".//t/s")]


@pytest.mark.parametrize("add_hydrogens", [True, False])
@pytest.mark.parametrize("smiles, expected", [("[H]", ["H"]), ("[2H]", ["D"])])
def test_hydrogen_and_deuterium_labels(smiles, expected, add_hydrogens):
    assert _labels(_cdxml(smiles, add_hydrogens=add_hydrogens)) == expected


@pytest.mark.parametrize("smiles, expected", [("N", "NH3"), ("O", "OH2")])
def test_labels_with_hydrogen_counts(smiles, expected):
    assert expected in _labels(_cdxml(smiles, add_hydrogens=True))


@pytest.mark.parametrize("smiles, expected, unexpected", [("N", "N", "NH"), ("O", "O", "OH")])
def test_labels_without_hydrogen_counts(smiles, expected, unexpected):
    labels = _labels(_cdxml(smiles, add_hydrogens=False))
    assert expected in labels
    assert unexpected not in labels


@pytest.mark.parametrize("smiles, expected", [("[NH4+]", ["N+"]), ("[OH-]", ["O-"])])
def test_charged_labels_without_hydrogen_counts(smiles, expected):
    assert _labels(_cdxml(smiles, add_hydrogens=False)) == expected


def test_num_hydrogens_is_total_count_without_hydrogen_labels():
    root = ET.fromstring(_cdxml("N", add_hydrogens=False).encode("utf-8"))
    assert [n.get("NumHydrogens") for n in root.iter("n")] == ["3"]


def test_aromatic_bonds_without_kekulize():
    assert 'Order="1.5"' in _cdxml("c1ccccc1", add_hydrogens=False, kekulize_molecule=False)


def test_kekulized_bonds():
    assert 'Order="2"' in _cdxml("c1ccccc1", add_hydrogens=False, kekulize_molecule=True)


def test_add_hydrogens_true_is_default():
    assert _cdxml("N") == _cdxml("N", add_hydrogens=True)


@pytest.mark.parametrize("smiles", ["c1ccccc1", "C[C@H](N)C(=O)O", "[NH4+]"])
def test_input_molecule_is_not_changed(smiles):
    mol = Chem.MolFromSmiles(smiles)
    before = (Chem.MolToSmiles(mol), mol.GetNumConformers(), [b.GetBondType() for b in mol.GetBonds()],
              [a.GetIsAromatic() for a in mol.GetAtoms()])
    mol_to_document(mol, add_hydrogens=False, kekulize_molecule=True)
    after = (Chem.MolToSmiles(mol), mol.GetNumConformers(), [b.GetBondType() for b in mol.GetBonds()],
             [a.GetIsAromatic() for a in mol.GetAtoms()])
    assert after == before


def test_existing_conformer_is_kept_without_hydrogen_labels():
    mol = Chem.MolFromSmiles("CCO")
    conformer = Chem.Conformer(mol.GetNumAtoms())
    for idx, x in enumerate([0.0, 1.5, 6.0]):
        conformer.SetAtomPosition(idx, (x, 0.0, 0.0))
    mol.AddConformer(conformer)
    root = ET.fromstring(mol_to_document(mol, add_hydrogens=False).to_cdxml().encode("utf-8"))
    xs = [float(n.get("p").split()[0]) for n in root.iter("n")]
    ys = [float(n.get("p").split()[1]) for n in root.iter("n")]
    assert len(set(ys)) == 1
    assert (xs[2] - xs[1]) / (xs[1] - xs[0]) == pytest.approx(3.0, rel=0.01)


def test_add_chiral_hs_adds_hydrogens_at_bridgeheads():
    smiles = "[C@@H]12CC[C@@H](CC1)C2"
    assert _cdxml(smiles) != _cdxml(smiles, add_chiral_hs=True)
    root = ET.fromstring(_cdxml(smiles, add_chiral_hs=True).encode("utf-8"))
    assert len(list(root.iter("n"))) == Chem.MolFromSmiles(smiles).GetNumAtoms() + 2


def test_add_wedge_bonds():
    smiles = "C[C@H](N)C(=O)O"
    assert "WedgeBegin" in _cdxml(smiles) or "WedgedHashBegin" in _cdxml(smiles)
    without_wedges = _cdxml(smiles, add_wedge_bonds=False)
    assert "WedgeBegin" not in without_wedges
    assert "WedgedHashBegin" not in without_wedges
