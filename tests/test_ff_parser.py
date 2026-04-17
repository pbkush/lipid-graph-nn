import pytest
import sys
import os
import json
from unittest.mock import patch

# Standard project testing insert block
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lipid_gnn.ff_parser import parse_martini_itp, create_ff_mapping


@pytest.fixture
def mock_itp_file(tmp_path):
    """
    Generate a mocked truncated .itp file containing the key physical parameter structures
    found in the Martini 3 configurations.
    """
    itp_content = """
;;;;; Martini 3 Force Field: Particle definition and LJ interactions
[ defaults ]
1 2 ; sigma-epsilon format of LJ parameters

[ atomtypes ]
; name mass charge ptype sigma epsilon
P6 72.0 0.000 A 0.0 0.0
C1 47.0 0.000 A 0.0 0.0
Q5 72.0 1.000 A 0.0 0.0

[ nonbond_params ]
; i j func sigma epsilon
P6 P6 1 4.700000e-01 4.990000e+00
P6 C1 1 4.700000e-01 3.520000e+00
C1 C1 1 4.700000e-01 3.390000e+00
Q5 Q5 1 4.700000e-01 6.450000e+00
"""
    file_path = tmp_path / "mock_martini.itp"
    with open(file_path, "w") as f:
        f.write(itp_content)
    return file_path


def test_parse_martini_itp(mock_itp_file):
    """
    Validate that the parser correctly extracts baseline category stats (mass, charge)
    and purely self-interaction nonbonded LJ parameters (sigma, epsilon).
    """
    ff_params = parse_martini_itp(mock_itp_file)
    
    # Test standard polar
    assert "P6" in ff_params
    assert ff_params["P6"]["mass"] == 72.0
    assert ff_params["P6"]["charge"] == 0.0
    assert ff_params["P6"]["sigma"] == 0.47
    assert ff_params["P6"]["epsilon"] == 4.99

    # Test variable apolar sizes
    assert "C1" in ff_params
    assert ff_params["C1"]["mass"] == 47.0
    assert ff_params["C1"]["sigma"] == 0.47
    assert ff_params["C1"]["epsilon"] == 3.39

    # Test explicit charged particles
    assert "Q5" in ff_params
    assert ff_params["Q5"]["charge"] == 1.0
    assert ff_params["Q5"]["epsilon"] == 6.45


def test_create_ff_mapping_execution(tmp_path):
    """
    Validate that the orchestration function can seamlessly load an integration 
    target and dump the valid mapping to JSON.
    """
    output_json = tmp_path / "test_out.json"
    
    # Mocking Path.exists to skip the hardcoded file structure verification 
    # and feeding a hardcoded parse return block
    with patch("lipid_gnn.ff_parser.Path.exists", return_value=True), \
         patch("lipid_gnn.ff_parser.parse_martini_itp", return_value={"P6": {"mass": 72.0}}) as mock_parse:
        
        create_ff_mapping("dummy_dir", output_json)
        
        assert mock_parse.called
        assert output_json.exists(), "JSON output file was not successfully created."
        
        # Verify correct formatting
        with open(output_json, "r") as f:
            data = json.load(f)
            
        assert "P6" in data
        assert data["P6"]["mass"] == 72.0
