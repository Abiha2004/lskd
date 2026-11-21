# Building the report

1. Copy your Mid-report images into this folder: `Picture1.png`, `Raw Pictures in CIFAR.png`.
2. Copy app screenshots to `screenshots/app_input.png` and `screenshots/app_predictions.png`.
3. From the repo root: `python scripts/make_report_assets.py` (creates `generated/` and `figures/`).
4. Upload this whole `report/` folder to Overleaf (or run `latexmk -pdf final_report.tex`).
5. Search the PDF for red **[TODO]** markers — none may remain in the submitted version.
