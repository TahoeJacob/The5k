"""
print_shop_pdf.py — minimal one-page summary PDF for the SLM print shop.

Side-view contour + a labelled channel cross-section + the smallest
features being asked for.  Nothing else.
"""
# --- run-from-anywhere shim (file lives in subfolder) ---
import os, sys
_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
os.chdir(_PARENT)
# --------------------------------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle

from main import config
from cea_interface import get_cea_for_analysis
from geometry import size_engine, build_contour


OUT_PDF = os.path.join(_PARENT, 'exports', 'print_shop_summary.pdf')


def main():
    cea  = get_cea_for_analysis(config)
    geom = size_engine(config, cea)

    L_c     = geom.L_c
    L_total = L_c + geom.L_nozzle
    t_w     = 1.0e-3                # wall thickness (constant)
    chan_w  = 1.0e-3                # constant 1 mm
    h_throat = config.chan_h_throat
    N_throat = config.N_channels_throat

    # Smallest land (rib between channels) sits at the throat
    land_throat = (2*np.pi*geom.R_t / N_throat) - chan_w

    # ---------- contour ----------
    x_arr, r_inner = build_contour(geom, dx=2e-4)
    r_outer = r_inner + t_w
    M = 1000.0
    x_mm = (x_arr - L_c) * M

    os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)

    with PdfPages(OUT_PDF) as pdf:
        fig = plt.figure(figsize=(11.69, 8.27))   # A4 landscape
        gs  = fig.add_gridspec(2, 2, height_ratios=[1.5, 1.2],
                               width_ratios=[1.2, 1.0],
                               hspace=0.45, wspace=0.25,
                               left=0.07, right=0.95, top=0.92, bottom=0.07)

        fig.suptitle('2.5 kN LOX/RP-1 Thrust Chamber — SLM print enquiry',
                     fontsize=14, fontweight='bold', y=0.97)

        # ============ Side-view contour (top, full width) ============
        ax1 = fig.add_subplot(gs[0, :])
        ax1.fill_between(x_mm,  r_outer*M,  r_inner*M, color='#7d7d7d')
        ax1.fill_between(x_mm, -r_outer*M, -r_inner*M, color='#7d7d7d')
        ax1.plot(x_mm,  r_inner*M, 'k-', lw=1.0)
        ax1.plot(x_mm, -r_inner*M, 'k-', lw=1.0)
        ax1.plot(x_mm,  r_outer*M, 'k-', lw=0.8)
        ax1.plot(x_mm, -r_outer*M, 'k-', lw=0.8)

        ax1.axvline(0.0, color='r', ls='--', lw=0.8)
        ax1.set_xlabel('Axial position (throat at 0) [mm]')
        ax1.set_ylabel('Radius [mm]')
        ax1.set_title('Engine cross-section (axisymmetric, hot wall shown in grey)',
                      fontsize=10, pad=10)
        ax1.set_aspect('equal')
        ax1.grid(alpha=0.3)
        y_max = float(np.max(r_outer*M)) * 1.25
        ax1.set_ylim(-y_max, y_max)

        # ============ Features table (bottom-left) ============
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.axis('off')
        ax2.set_title('Smallest features — please confirm achievable',
                      fontsize=11, fontweight='bold', loc='left', pad=8)

        rows = [
            ['Hot-wall thickness', f'{t_w*M:.2f} mm'],
            ['Channel width',      f'{chan_w*M:.2f} mm'],
            ['Channel height',     f'{h_throat*M:.2f} mm  (at throat)'],
            ['Land / rib width',   f'{land_throat*M:.2f} mm  (at throat)'],
            ['Channel length',     f'{L_total*M:.0f} mm  (continuous)'],
            ['Number of channels', f'{N_throat} at throat'],
        ]
        tbl = ax2.table(cellText=rows, colWidths=[0.48, 0.52],
                        loc='upper left', cellLoc='left', edges='horizontal')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1.0, 1.55)
        for (i, j), cell in tbl.get_celld().items():
            cell.set_edgecolor('#888888')
            if j == 1:
                cell.set_text_props(weight='bold', color='#b00000')

        # ============ Channel cross-section (bottom-right) ============
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.set_title('Channel cross-section (at throat)',
                      fontsize=11, fontweight='bold', loc='left', pad=8)

        # Draw two cells so the land between channels is visible
        cw  = chan_w * M        # 1.0
        ch  = h_throat * M      # 1.10
        ld  = land_throat * M   # 0.93
        tw  = t_w * M           # 0.90
        cls = 1.2               # closeout slab thickness (cosmetic)

        # Layout: [half-land | channel | land | channel | half-land]
        x0 = 0.0
        segs = [
            ('wall',    x0,              ld/2 + cw + ld + cw + ld/2),  # full width slab
        ]
        total_w = ld/2 + cw + ld + cw + ld/2

        # Hot wall slab
        ax3.add_patch(Rectangle((0, 0), total_w, tw,
                                facecolor='#7d7d7d', edgecolor='k', lw=1.0))
        # Left half-land
        ax3.add_patch(Rectangle((0, tw), ld/2, ch,
                                facecolor='#7d7d7d', edgecolor='k', lw=1.0))
        # Left channel
        ax3.add_patch(Rectangle((ld/2, tw), cw, ch,
                                facecolor='#cde3f5', edgecolor='k', lw=1.0))
        # Middle land
        ax3.add_patch(Rectangle((ld/2 + cw, tw), ld, ch,
                                facecolor='#7d7d7d', edgecolor='k', lw=1.0))
        # Right channel
        ax3.add_patch(Rectangle((ld/2 + cw + ld, tw), cw, ch,
                                facecolor='#cde3f5', edgecolor='k', lw=1.0))
        # Right half-land
        ax3.add_patch(Rectangle((ld/2 + cw + ld + cw, tw), ld/2, ch,
                                facecolor='#7d7d7d', edgecolor='k', lw=1.0))
        # Closeout slab
        ax3.add_patch(Rectangle((0, tw + ch), total_w, cls,
                                facecolor='#bfbfbf', edgecolor='k', lw=1.0))

        # Axis labels for orientation
        ax3.annotate('HOT GAS', (total_w/2, -0.25), ha='center',
                     fontsize=9, color='#b00000', fontweight='bold')
        ax3.annotate('COOLANT JACKET', (total_w/2, tw + ch + cls + 0.15),
                     ha='center', fontsize=9, color='#003080', fontweight='bold')

        # Leader-line callouts — all point outward to the right/left, no
        # arrows over the geometry itself.
        def callout(xy_start, xy_text, text, color='#b00000'):
            ax3.annotate(text, xy=xy_start, xytext=xy_text,
                         fontsize=9, color=color, ha='left', va='center',
                         arrowprops=dict(arrowstyle='-', color=color, lw=1.2))

        # Hot-wall thickness — leader from left edge of wall slab
        callout((0.0, tw/2),     (-1.6, tw/2),
                f'Hot-wall thickness\n{tw:.2f} mm')
        # Channel width — leader from top of left channel
        callout((ld/2 + cw/2, tw + ch),  (-1.6, tw + ch + 0.3),
                f'Channel width\n{cw:.2f} mm')
        # Channel height — leader from right side of right channel
        callout((ld/2 + cw + ld + cw, tw + ch/2),  (total_w + 0.4, tw + ch/2),
                f'Channel height\n{ch:.2f} mm')
        # Land width — leader from top of middle land
        callout((ld/2 + cw + ld/2, tw + ch),  (total_w + 0.4, tw + ch + 0.35),
                f'Land (rib)\n{ld:.2f} mm')

        ax3.set_xlim(-2.8, total_w + 2.8)
        ax3.set_ylim(-0.8, tw + ch + cls + 0.8)
        ax3.set_aspect('equal')
        ax3.axis('off')

        pdf.savefig(fig)
        plt.close(fig)

    print(f"\nWrote {OUT_PDF}")


if __name__ == "__main__":
    main()
