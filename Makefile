# Retriever module — routine commands (retriever-module-design.md §6)

retriever-up:
	docker compose -f infra/qdrant/docker-compose.yml up -d --wait
	uv run scripts/retriever_ctl.py check

retriever-schema:
	uv run scripts/retriever_ctl.py apply-schema

retriever-status:
	uv run scripts/retriever_ctl.py status

retriever-down:
	docker compose -f infra/qdrant/docker-compose.yml down
