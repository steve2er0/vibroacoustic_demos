#!/usr/bin/env python3
"""Compute pseudo PSD / CPSD from replayed complex harmonic response CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

from force_recon.response_spectra import (
    auto_psd_from_complex_response,
    infer_uniform_df_hz,
    load_complex_response_csv,
    spectral_matrix_from_complex_response,
    write_auto_psd_csv,
    write_cpsd_csv,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Compute auto-PSD and CPSD estimates from a complex harmonic response CSV "
            "with columns freq_hz,re0,im0,re1,im1,..."
        )
    )
    p.add_argument("response_csv", help="Complex response CSV path.")
    p.add_argument(
        "--auto-out",
        help="Output CSV for auto PSD channels. Defaults to <input>_auto_psd.csv",
    )
    p.add_argument(
        "--cpsd-out",
        help="Output CSV for upper-triangle CPSD terms. Defaults to <input>_cpsd.csv",
    )
    p.add_argument(
        "--df-hz",
        type=float,
        default=None,
        help="Frequency spacing in Hz. If omitted, infer from the frequency vector.",
    )
    p.add_argument(
        "--phasor-convention",
        choices=["peak", "rms"],
        default="peak",
        help=(
            "Interpret the complex response magnitudes as peak or RMS amplitudes. "
            "Use 'peak' for typical SOL111 complex response output."
        ),
    )
    return p


def default_output_path(input_path: Path, suffix: str) -> Path:
    return input_path.with_name(f"{input_path.stem}{suffix}")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    input_path = Path(args.response_csv)
    auto_out = Path(args.auto_out) if args.auto_out else default_output_path(input_path, "_auto_psd.csv")
    cpsd_out = Path(args.cpsd_out) if args.cpsd_out else default_output_path(input_path, "_cpsd.csv")

    freqs_hz, response_complex, channel_names = load_complex_response_csv(input_path)
    df_hz = float(args.df_hz) if args.df_hz is not None else infer_uniform_df_hz(freqs_hz)
    auto_psd = auto_psd_from_complex_response(
        response_complex, df_hz=df_hz, phasor_convention=args.phasor_convention
    )
    cpsd = spectral_matrix_from_complex_response(
        response_complex, df_hz=df_hz, phasor_convention=args.phasor_convention
    )

    write_auto_psd_csv(auto_out, freqs_hz, auto_psd, channel_names)
    write_cpsd_csv(cpsd_out, freqs_hz, cpsd, channel_names)

    print(f"Input: {input_path}")
    print(f"Channels: {len(channel_names)}")
    print(f"Frequency lines: {len(freqs_hz)}")
    print(f"df_hz: {df_hz:.16g}")
    print(f"Phasor convention: {args.phasor_convention}")
    print(f"Auto PSD CSV: {auto_out}")
    print(f"CPSD CSV: {cpsd_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
