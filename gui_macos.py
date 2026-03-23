"""
CLI / headless Matplotlib plots (``--save`` or path-based run).

Backend: ``macosx`` on Darwin, ``TkAgg`` elsewhere (Windows/Linux).

``--pick`` uses ``gui_file_dialogs`` (AppleScript on macOS, Tk dialog on Windows).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

if sys.platform == "darwin":
    matplotlib.use("macosx")
else:
    matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np

from force_recon import frf_io, pipeline, units
from force_recon.export_nastran import write_force_spectrum_csv
from force_recon.flight_io import load_flight_data
from gui_file_dialogs import choose_open_file, choose_open_files, choose_save_file
from gui_plotting import draw_conditioning, draw_spectra


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Force reconstruction — CLI plots or --save (no interactive time slice)"
    )
    p.add_argument(
        "--pick",
        action="store_true",
        help="Choose Fx/Fy/Fz CSVs and flight CSV/MAT files via native dialogs",
    )
    p.add_argument("--fx", type=Path, help="Mobility CSV unit Fx")
    p.add_argument("--fy", type=Path, help="Mobility CSV unit Fy")
    p.add_argument("--fz", type=Path, help="Mobility CSV unit Fz")
    p.add_argument(
        "--ones-h",
        action="store_true",
        help="Use a dummy mobility matrix H=1+0j derived from the flight timebase",
    )
    flight_group = p.add_mutually_exclusive_group()
    flight_group.add_argument("--flight", type=Path, help="Flight CSV (time_s + channels in g)")
    flight_group.add_argument(
        "--flight-mat",
        type=Path,
        nargs="+",
        help="Ordered per-channel MAT files with amp/t/sr fields",
    )
    p.add_argument("--t0", type=float, default=0.0)
    p.add_argument("--t1", type=float, default=1.0)
    p.add_argument("--g0", type=float, default=units.G0_STANDARD)
    p.add_argument("--lam", type=float, default=0.0, help="Tikhonov λ")
    p.add_argument("--mobility-si", action="store_true", help="H in (m/s)/N not imperial")
    p.add_argument("--channel", type=int, default=0, help="Channel index for accel PSD")
    p.add_argument("--save", type=Path, help="Write F̂ CSV to this path and exit (no plot)")
    args = p.parse_args(argv)

    fx, fy, fz = args.fx, args.fy, args.fz
    flight_input: Path | list[Path] | None = args.flight if args.flight is not None else args.flight_mat
    if args.pick:
        if not args.ones_h:
            print("Choose Fx mobility CSV…")
            fx = choose_open_file("Choose Fx mobility CSV")
            print("Choose Fy mobility CSV…")
            fy = choose_open_file("Choose Fy mobility CSV")
            print("Choose Fz mobility CSV…")
            fz = choose_open_file("Choose Fz mobility CSV")
        print("Choose flight CSV or MAT file(s)…")
        flight_paths = choose_open_files("Choose flight CSV or per-channel MAT files")
        if len(flight_paths) == 1:
            flight_input = flight_paths[0]
        elif len(flight_paths) > 1:
            flight_input = flight_paths
        else:
            flight_input = None
        if (not args.ones_h and not all([fx, fy, fz])) or flight_input is None:
            print("Cancelled.", file=sys.stderr)
            return 1
    elif ((not args.ones_h and not all([fx, fy, fz])) or flight_input is None):
        p.error(
            "Provide --flight or --flight-mat plus either --ones-h or --fx --fy --fz, or use --pick"
        )

    assert flight_input is not None

    try:
        time_s, acc_g, ch_names = load_flight_data(flight_input)
        if args.ones_h:
            frf_f, H_imp = frf_io.build_ones_mobility(time_s, acc_g.shape[1])
            mobility_si = True
        else:
            assert fx is not None and fy is not None and fz is not None
            frf_f, H_imp = frf_io.load_mobility_csv_triplet(fx, fy, fz)
            mobility_si = args.mobility_si
    except Exception as e:
        print(f"Load error: {e}", file=sys.stderr)
        return 1

    if acc_g.shape[1] != H_imp.shape[1]:
        print(
            f"Channel mismatch: FRF {H_imp.shape[1]} vs flight {acc_g.shape[1]}",
            file=sys.stderr,
        )
        return 1

    ch_i = max(0, min(args.channel, acc_g.shape[1] - 1))

    try:
        res = pipeline.reconstruct_forces(
            time_s=time_s,
            acc_g=acc_g,
            t_start=args.t0,
            t_end=args.t1,
            frf_freqs_hz=frf_f,
            H_imperial=H_imp,
            g0=args.g0,
            tikhonov_lambda=args.lam,
            mobility_is_si=mobility_si,
        )
    except Exception as e:
        print(f"Run error: {e}", file=sys.stderr)
        return 1

    if args.save:
        write_force_spectrum_csv(
            res.freqs_hz, res.F_hat, args.save, valid_mask=res.valid_mask
        )
        print(f"Wrote {args.save}")
        return 0

    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    fig1.canvas.manager.set_window_title("Force reconstruction — spectra")
    draw_spectra(ax1, ax2, res, time_s, acc_g, ch_i, ch_names, args.t0, args.t1, args.g0)
    fig1.tight_layout()

    fig2, axs = plt.subplots(2, 2, figsize=(10, 7))
    fig2.canvas.manager.set_window_title("Force reconstruction — conditioning")
    draw_conditioning([axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]], res)
    fig2.tight_layout()

    def on_key(event):
        if event.key in ("s", "S"):
            out = choose_save_file("F_hat_spectrum.csv")
            if out:
                write_force_spectrum_csv(
                    res.freqs_hz, res.F_hat, out, valid_mask=res.valid_mask
                )
                print(f"Saved {out}")

    fig1.canvas.mpl_connect("key_press_event", on_key)
    fig2.canvas.mpl_connect("key_press_event", on_key)
    print("Close plot windows when done. Press 's' in a plot window to save F̂ CSV.")
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
