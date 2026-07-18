.PHONY: behaviour-venv behaviour-download behaviour-validate behaviour-features behaviour-test

behaviour-venv:
	cd ml/behaviour && python -m venv .venv

behaviour-download:
	cd ml/behaviour && .venv/Scripts/python src/download_data.py

behaviour-validate:
	cd ml/behaviour && .venv/Scripts/python src/validate_studentlife.py

behaviour-features:
	cd ml/behaviour && .venv/Scripts/python src/build_daily_features.py

behaviour-test:
	cd ml/behaviour && .venv/Scripts/python -m pytest tests/ -v
