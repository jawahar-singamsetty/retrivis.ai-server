# Makefile - ADD THIS
.PHONY: server worker redis eval-collect eval-run eval-full

# Development servers
server:
	poetry run uvicorn src.server:app --reload --host 0.0.0.0 --port 8000

worker:
	poetry run celery -A src.services.celery:celery_app worker --loglevel=info --pool=threads -E

redis:
	wsl bash start_redis.sh

# Stop services
stop:
	wsl bash stopAll.sh

# Evaluation tasks
eval-collect:
	poetry run python evaluation/scripts/raga_data_collection.py

eval-run:
	poetry run python evaluation/scripts/run_evaluation.py

eval-full: eval-collect eval-run
	@echo "✅ Evaluation complete!"