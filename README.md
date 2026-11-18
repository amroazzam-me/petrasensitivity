# Petra pipelines: closed-form sensitivity analysis

Analysis code for the manuscript *Hydraulic and Kinematic Systems in Historical
Jordan: Three Case Studies from the Nabataean to the Ottoman Period*
(Amro Azzam, Department of Mechanical Engineering, Mutah University).

The code reanalyses the published Siq and Wadi Mataha pipeline inputs of
Ortloff (2020), *Water* 12(12), 3498, with a closed-form partly full circular
pipe model (critical depth, Colebrook-White friction, ratio S/Sc). It is neither
a replication of Ortloff's CFD nor an independent validation of it.

## Files
- `petra_sensitivity.py`: baseline checks, one-at-a-time sensitivity and Monte Carlo (N = 4000); produces manuscript Figures 1-3.
- `petra_extended.py`: five range scenarios (N = 40,000), Sobol indices, band dependence, Appendix A-B tables and Figures 4, 5, B1. *(add this file)*

## Run
Python 3.12 or later is required (nested quotes inside f-strings).

    pip install -r requirements.txt
    python petra_sensitivity.py
    python petra_extended.py 40000

Outputs are written to `./petra_outputs/`. Each input is labelled PUBLISHED,
INFERRED or ASSUMED in the script and in Appendix A of the manuscript.

## Contact
Amro Azzam, azzamamro787@gmail.com
