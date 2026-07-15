PY := ./venv/bin/python

.DEFAULT_GOAL := help

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

models:  ## Rebuild the 3 ML models (writes outputs/metrics/*.json)
	$(PY) src/models/churn_model.py
	$(PY) src/models/segmentation_model.py
	$(PY) src/models/forecasting_model.py

dbt:  ## Build dbt models + run data-quality tests
	cd dbt && ../venv/bin/dbt build --profiles-dir .

report:  ## Generate the grounding-verified LLM executive report (needs GROQ_API_KEY)
	$(PY) src/llm/grounded_report.py

data:  ## Export dashboard data to web/data/*.json
	$(PY) scripts/export_dashboard_data.py

test:  ## Run the Python test suite
	$(PY) -m pytest tests/ -q

dashboard:  ## Install + build the Next.js dashboard
	cd web && npm install && npm run build

all: models dbt report data test  ## Full pipeline: models -> dbt -> report -> data -> tests
	@echo "\n✅ RevenueIQ pipeline complete — all metrics reproduced and verified."

.PHONY: help models dbt report data test dashboard all
