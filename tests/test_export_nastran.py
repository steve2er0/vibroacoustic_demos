"""Tests for NASTRAN-oriented text export helpers."""

import numpy as np

from force_recon.export_nastran import tabled1_re_im_snippets, tabrnd1_snippet


def test_tabled1_re_im_snippets_use_free_field_format():
    freqs_hz = np.array([10.0, 11.0])
    force = np.array([1.25 - 2.5j, -3.0 + 4.75j])

    text = tabled1_re_im_snippets(1001, 1002, freqs_hz, force)
    lines = text.splitlines()

    assert lines[0] == "TABLED1,1001,LINEAR,LINEAR"
    assert lines[1] == "+,1.0000000000000000E+01,1.2500000000000000E+00"
    assert lines[2] == "+,1.1000000000000000E+01,-3.0000000000000000E+00,ENDT"
    assert lines[3] == "TABLED1,1002,LINEAR,LINEAR"
    assert lines[4] == "+,1.0000000000000000E+01,-2.5000000000000000E+00"
    assert lines[5] == "+,1.1000000000000000E+01,4.7500000000000000E+00,ENDT"


def test_tabrnd1_snippet_use_free_field_format():
    freqs_hz = np.array([10.0, 20.0])
    psd = np.array([0.125, 0.5])

    text = tabrnd1_snippet(2001, freqs_hz, psd)
    lines = text.splitlines()

    assert lines[0] == "TABRND1,2001,1,1"
    assert lines[1] == "+,1.0000000000000000E+01,1.2500000000000000E-01"
    assert lines[2] == "+,2.0000000000000000E+01,5.0000000000000000E-01,ENDT"
