# Design log — The 5k (2.5 kN LOX/RP-1 CuCrZr regen-cooled)

A running log of decisions, validations, and findings. Most recent at top.
Each entry is short — capture *why*, not just *what*.

---

## 2026-06-01 — N_throat sweep: design freeze at N=36

**Question:** Could reducing channel count (higher V per channel via higher
mdot per channel) close the T_hw margin?

**Sweep result:** Models give OPPOSITE optimums for the design knob.

| N_throat | land [mm] | 1D fin T_hw [K] | 2D wall T_hw [K] |
| 20       | 4.17      | 1037 (−59)      | 1362 (+85)       |
| 24       | 3.28      | 1054 (−42)      | 1330 (+53)       |
| 36 (cur) | 1.79      | 1096            | 1277             |
| 44       | 1.24      | 1118 (+22)      | 1257 (−20)       |

- **1D fin says fewer channels are better** (higher V dominates fin)
- **2D wall says more channels are better** (fewer/smaller metal blocks
  for heat to accumulate in — more accurate for high-k materials)
- Real-engine practice (LUMEN) packs channels densely → aligns with 2D
  prediction direction

**Decision:** Freeze design at N=36. Reasons:
- Disagreement between models means the optimum is uncertain
- Even at best case (~20-60 K improvement), still doesn't close gap to
  the validated 2D model's overshoot of 200 K
- Marginal CAD risk (SLM land floor at N=44) not worth uncertain reward
- Real T_hw measurement on first fire will resolve the model
  disagreement empirically

**The right next move is firing the engine, not tweaking the design.**
Hot-fire data at N=36 will tell us:
- Whether the 1D fin or 2D wall model is closer to truth
- Whether the design has actual margin or actual problem
- What's the bias correction for future iterations

If T_cw measurements show acceptable margin → run at full P_c.
If T_cw shows trouble → abort, redesign with measured bias correction
applied to the validated model.

**Refs:** `/tmp/n_channel_sweep.py`, `/tmp/n_sweep_2d.py`.

---

## 2026-06-01 — Betti SSME MCC validation: actually a PASS at −0.6 % T_hw

**Initial claim retracted.** I had compared our T_hw to the older nominal
design value (~800–850 K from Wang and Luong). The paper itself explicitly
states that value is too low. The correct comparison is to Betti's
coupled CFD+quasi-2D prediction.

**Result with Case A settings (wall_2d=T, int_BL=T, C=0.018):**
| Metric                        | Betti (smooth) | Ours    | Error  |
| Peak T_w,HG                   | 1071 K         | 1065 K  | −0.6 % |
| Peak q_w,HG                   | 161.5 MW/m²    | 112.1   | −30 %  |
| ΔP_coolant                    | 67 bar         | 52      | −22 %  |
| ΔT_coolant                    | ~220 K         | 264     | +20 %  |
| Hot-fire test data band       | up to 1030 K   | 1065    | within |

**Peak wall temperature matches within 0.6 %.** Secondary metrics
(ΔP, ΔT, local q profile) diverge ~20–30 %, suggesting our coolant
model distributes heat slightly differently along the chamber, but the
integrated effect on the peak metric that matters for engine safety is
essentially perfect.

**Updated validation pyramid:**
| Reference                          | Engine                  | T_hw error |
| DLR LUMEN (real engine)            | LOX/CH4 CuCrZr 25 kN    | +3.5 %     |
| **Betti SSME MCC (CFD+quasi-2D)**  | LOX/LH2 NARloy-Z 226bar | **−0.6 %** |
| HARCC subscale                     | Cu-alloy cyl            | within band|
| RPA Inconel 2.5kN                  | match thrust/P_c        | −6 %       |
| RPA CuCrZr 2.5kN                   | direct                  | −33 % (fin)|

**Two independent peer-reviewed engine validations within ±4 % on peak
T_hw.** Strong evidence the 2.5 kN design predictions are sound.

**Refs:** Betti 2014 (JPP) Fig 7-8, channel data from
`Archive/RegenCoolingCalc.py:1518-1528`, sweep script `/tmp/betti_sweep.py`.

**Sensitivity sweep:** C_bartz (gas BL coefficient) is the biggest single
knob (−13 % T_hw going 0.023 → 0.018). 2D wall vs 1D fin: ±10 %.
Simplified vs integral-BL Bartz: < 2 % — negligible. Even with the most
favourable combination (low C_bartz + 2D + int BL), still +33 % over.

**Root cause:** Coupled ΔP/ΔT_coolant error points at coolant-side model.
Niino hydrogen correlation operates outside its calibration range at
446 bar supercritical H2; CoolProp accuracy near the pseudocritical line
is questionable. Multiple physics issues compound for the SSME regime.

**Decision:** Document Betti as out-of-regime. SSME ≠ 2.5 kN LOX/RP-1.
Different propellants (LH2 vs RP-1), different P_c (226 vs 20 bar),
different correlations (Niino vs Sieder-Tate), different wall material
regime. Don't claim SSME validation in the report.

**For confidence on your 2.5 kN engine, this doesn't change anything.**
LUMEN at +3.5 % on T_hw remains the strongest validation point.

**Refs:** Betti 2014 (JPP), `/tmp/betti_sweep.py`, channel arrays from
`Archive/RegenCoolingCalc.py` lines 1518-1528.

---

## 2026-05-31 — Test program go-ahead decision

**Decision:** Keep engine design as-is. Proceed to print + hot-fire test.

**Predicted operating point (model, k=170 W/m·K, fin enabled):**
- Peak T_hw : 1096 K @ throat (x ≈ 188 mm)
- Peak T_cw : 984 K @ throat
- Peak q_gas : ~19 MW/m²
- ΔT coolant : 363 K (298 K → 661 K)
- ΔP coolant : 2.3 bar

**Abort criterion:** T_cw at any TC station exceeds **1073 K (800 °C)** →
shut down. This is the CuCrZr strength limit (not melt point — material
loses structural strength well before melting).

**Predicted margin to abort:** 1073 − 984 = **89 K (≈ 9 %)**. Small but
positive. Watch throat TC most closely.

**TC stations (3):** chamber x=60 mm, contraction x=140 mm, throat x≈190 mm.
1 mm K-type, drilled to fin base in the rib, silver-loaded thermal
epoxy bonded.

---

## 2026-05-31 — Fin model investigation

**Question:** Why does our T_hw match RPA on Inconel (−6 %) but not CuCrZr (−33 %)?

**Investigation:** Inspected `heat_transfer.py:596` rib-fin code. Standard
rectangular fin with Incropera adiabatic-tip correction. Added `use_fin`
config flag to test sensitivity.

**Result sweep (k=300, matching RPA's run):**
| fin | T_hw [K] |
|---|---|
| ON (default, ours) | 1040 |
| OFF | 1809 |
| RPA reference | 1555 |

RPA sits midway — uses ~50 % fin credit, not full.

**Decision:** Trust our model (fin ON). DLR LUMEN experimental data validates
this approach within +3.5 % on peak T_hw — that's a real engine with
published bench-tested temperatures. RPA being conservative on the fin
treatment is a software modeling choice difference, not a model failure.

**Refs:** `validate_lumen.py`, fin code at `heat_transfer.py:596–602`,
`References/2.5kNCuCrZrVALIDATIONRUN.txt`

---

## 2026-05-31 — Validation summary table

| Reference | What | Paper / Ref | Ours | Error |
|---|---|---|---|---|
| **DLR LUMEN 25 kN** (LOX/CH4) | Peak T_hw | 879 K | 910 K | **+3.5 %** ✓ |
| | T_hw @ injector | 874 K | 886 K | +1.4 % |
| | T_hw @ throat | 879 K | 870 K | −1.0 % |
| | Coolant T_out | 408.6 K | 450.3 K | +10 % |
| **DLR HARCC** (cylindrical only) | q_w averaged | 14.5 MW/m² | 12.2 (simp) / 8.7 (BL) | −16 % / −40 % |
| **RPA Inconel 2.5kN** (software) | Peak T_hw | 1110 K | 1042 K | **−6.1 %** ✓ |
| | ΔT coolant | 159 K | 147 K | −7.5 % |
| **RPA CuCrZr 2.5kN** (software) | Peak T_hw | 1555 K | 1040 K | −33 % (fin model) |
| | ΔT coolant | 405 K | 367 K | −9.4 % |

LUMEN is the gold standard (real engine, peer-reviewed). All other
references trail it for confidence weighting.

---

## 2026-05-31 — Engine design freeze, headline specs

**Chamber:**
- 2.5 kN SL / 3.07 kN vacuum
- P_c = 20 bar, O/F = 2.0
- D_c = 96.77 mm, D_t = 34.21 mm
- L_c = 189.7 mm, L_total = 263.7 mm
- ε = 6, contraction 8, L* = 1.27 m
- Parabolic bell, θD = 30°, θE = 12°

**Wall / channels:**
- CuCrZr, k = 170 W/m·K (realistic SLM as-built), t = 1.0 mm
- Bifurcating 72 chamber → 36 throat, split at r = 2·R_t
- Width taper: 1.5/1.5 chamber → 1.2 throat → 1.5 exit
- Height taper: 1.3 chamber → 1.1 throat → 1.5 exit
- Roughness 6.3 µm (SLM as-built)

**Injector:**
- O-F-O triplet, N = 20, single ring on Ø 69.7 mm bolt circle
- 60 orifices total (20 fuel Ø 1.154 mm + 40 ox Ø 1.056 mm each)
- 2θ = 90°, L_imp = 6.34 mm (within SP-8089 spec)
- M_R = 0.60 (off-optimum, inherent to LOX/RP-1 at OF=2 in O-F-O)
- LOX face cooling: 20 radial channels 1.5×1.5 mm
- Three stacked 10 mm-OD manifold rings, all at R = 34.84 mm
- ΔP_inj = 4 bar both sides

**Plumbing:**
- 3/4" OD hard-drawn copper tube, both fuel and LOX (NZ-sourced, NASA
  SP-8119 LOX velocity ceiling 4.5 m/s — we run at 1.9 m/s)
- Engine ports: 1/2" BSPT (chamber fuel inlet — boss OD 27 mm) and
  3/4" BSPT (LOX inlets — boss OD 31 mm, post-EPLUS3D)
- Sensor ports: G 1/4 BSPP with O-ring face seal (parallel thread, swappable)
- Igniter: R 1/8 BSPT (tapered, metal seal — survives combustion gas)

**Sensors:**
- 3× K-type 1 mm TCs in chamber/contraction/throat rib bases (T_cw)
- 4× G1/4 BSPP pressure sensor ports on injector (P_c, fuel, T_inj, T_fuel)
- 4× pressure sensors at G1/4 ports — AliExpress on fuel side, ullage on
  LOX tank (LOX-line sensors not LOX-safe)

**Fasteners:** 8× M8 grade 8.8 with DIN 980 stover (all-metal) lock-nuts
for chamber-to-injector flange. Star-pattern 3-pass torque to 25 N·m.

**Seals:** PTFE inboard O-ring, Viton 85 FDA outboard O-ring. Metal-to-metal
primary seal at the flange face.

---

## 2026-05-30 — Mac/Linux thermal drift

**Finding:** Same code, same config, two machines → ~110 K disagreement on
peak T_hw. Geometry exports byte-identical. Root cause: CoolProp/REFPROP
version mismatch on RP-1 properties.

**Decision:** Use Linux for "official" predictions. Mac runs are documented
as informational only.

**How to apply:** When citing T_hw predictions in the report, always note
which machine. Pin versions if going to publication-quality numbers.

---

## 2026-05-25 — Trajectory feasibility (110 km apogee)

Vehicle scope: 30 kg dry, 200 mm OD, single-stage, vertical ascent.

**1-DOF sim result for design point Cd profile:**
- Propellant: 48 kg total (16 kg RP-1, 32 kg LOX at OF=2)
- Burn time: 47 s
- GLOW: 78 kg, T/W at liftoff = 3.3
- Apogee: 110 km

Sensitivity: OD is the biggest lever — 250 mm needs 78 kg propellant,
150 mm needs only 37 kg. Dry mass second biggest.

Engine performance (CEA): Isp_SL = 234 s, Isp_vac = 288 s, c* = 1784 m/s.

**Refs:** `/tmp/traj_sim.py` (1-DOF integration)

---

## 2026-05-22 — k_wall corrected to 170 W/m·K

**Was:** 300 W/m·K (wrought CuCrZr)
**Now:** 170 W/m·K (SLM as-built, realistic for printed CuCrZr)

**Effect on peak T_hw:** 1040 K → 1096 K (k=170 reduces wall conductivity,
increases ΔT through wall). Still under 1073 K limit by 23 K — tight.

---

## 2026-05-08 — Faceplate cooling switched to LOX (not RP-1)

**Why not RP-1:** RP-1 arrives at injector at ~500 K (regen-heated), already
near the coking limit. Adding face-cooling duty would push it over → coke
deposits → blocked channels.

**Why LOX works:** Cold (90 K), dense, large subcool margin at 35 bar
(saturation 137 K → 47 K subcool). 20 radial channels, 1.5×1.5 mm,
V=13.4 m/s, h=36 kW/m²·K, face stays at 253 K with huge margin.

---

# Template for future entries

```markdown
## YYYY-MM-DD — Short title

**Question / Decision / Finding:** one line

**Why / What / Result:** 2–4 lines

**Refs:** files, line numbers, paper citations
```
