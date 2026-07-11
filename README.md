# Instanton Transition Engine for ETFs

Models regime shifts as quantum tunneling events between local minima of an effective potential. Instanton solutions (bounce configurations in imaginary time) compute transition amplitudes and expected arrival times of crashes or rallies. The per‑ETF score is the tunneling amplitude – high = likely regime shift.

## Features
- Three ETF universes (FI/Commodities, Equity Sectors, Combined)
- Seven rolling windows (63–4536 days)
- Double-well potential from returns distribution
- Instanton action via numerical integration
- Tunneling amplitude = exp(-S) / (1 + exp(-S))
- Macro variables modulate the potential landscape
- Two‑tab Streamlit dashboard (auto best, manual)
- Results stored on Hugging Face: `P2SAMAPA/p2-etf-instanton-transition-results`

## Usage

1. Set `HF_TOKEN` environment variable.
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python train.py` (fast)
4. Launch dashboard: `streamlit run streamlit_app.py`

## Interpretation

- High tunneling amplitude → likely regime shift (crash or rally).
- Low amplitude → stable regime.

## Requirements

See `requirements.txt`.
