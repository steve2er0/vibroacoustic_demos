"""Tests for NASTRAN-oriented text export helpers."""

import numpy as np

from force_recon.export_nastran import tabled1_re_im_snippets, tabrnd1_snippet


def test_tabled1_re_im_snippets_use_free_field_format():
    freqs_hz = np.array([10.0, 11.0])
    force = np.array([1.25 - 2.5j, -3.0 + 4.75j])

    text = tabled1_re_im_snippets(1001, 1002, freqs_hz, force)
    lines = text.splitlines()

    assert lines[0] == "TABLED1     1001  LINEAR  LINEAR"
    assert lines[1] == "             10.    1.25     11.     -3.    ENDT"
    assert lines[2] == "TABLED1     1002  LINEAR  LINEAR"
    assert lines[3] == "             10.    -2.5     11.    4.75    ENDT"


def test_tabrnd1_snippet_use_free_field_format():
    freqs_hz = np.array([10.0, 20.0])
    psd = np.array([0.125, 0.5])

    text = tabrnd1_snippet(2001, freqs_hz, psd)
    lines = text.splitlines()

    assert lines[0] == "TABRND1     2001"
    assert lines[1] == "             10.    .125     20.      .5    ENDT"


def test_tabled1_re_im_snippets_pack_full_continuation_lines():
    freqs_hz = np.array([10.0, 11.0, 12.0, 13.0])
    force = np.array([1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j, 4.0 + 0.0j])

    text = tabled1_re_im_snippets(1001, 1002, freqs_hz, force)
    lines = text.splitlines()

    assert lines[0] == "TABLED1     1001  LINEAR  LINEAR"
    assert lines[1] == "             10.      1.     11.      2.     12.      3.     13.      4."
    assert lines[2] == "            ENDT"
