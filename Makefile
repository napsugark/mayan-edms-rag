# ============================================================
# Makefile — Unified Mayan EDMS + Hybrid RAG + Langfuse Stack
#
# Usage:
#   make up          Start the full stack (default profiles)
#   make up-ollama   Start with local Ollama LLM
#   make down        Stop all services (keep volumes)
#   make restart     Restart all services
#   make destroy     Stop all services AND delete volumes
#   make logs        Follow logs for all services
#   make ps          Show running containers
#   make build       Rebuild images (rag)
#   make rag-shell   Open a shell in the RAG container
#   make mayan-shell Open a shell in the Mayan container
#   make index       Run the document indexing script
#   make backup-db   Backup Mayan PostgreSQL database
# ============================================================

COMPOSE       := docker compose
COMPOSE_FILE  := docker-compose.yml
ENV_FILE      := .env

# Default profiles are set in .env via COMPOSE_PROFILES
# Override here if needed:
#   make up PROFILES="all_in_one,postgresql,rabbitmq,redis,ollama"
PROFILES      ?=

ifdef PROFILES
  PROFILE_FLAGS := --profile $(subst $(,), --profile ,$(PROFILES))
else
  PROFILE_FLAGS :=
endif

.PHONY: help up up-ollama down restart destroy logs ps build \
        rag-shell mayan-shell index backup-db health

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

# ----------------------------------------------------------
# Lifecycle
# ----------------------------------------------------------

up: ## Start all services (default profiles from .env)
	$(COMPOSE) up -d --build

up-ollama: ## Start all services including Ollama
	$(COMPOSE) --profile ollama up -d --build

down: ## Stop all services (volumes preserved)
	$(COMPOSE) --profile "*" down

restart: ## Restart all services
	$(COMPOSE) --profile "*" down
	$(COMPOSE) up -d --build

destroy: ## Stop all services AND delete volumes (DATA LOSS!)
	@echo "WARNING: This will delete ALL data volumes!"
	@echo "Press Ctrl+C to cancel, or wait 5 seconds..."
	@sleep 5
	$(COMPOSE) --profile "*" down -v

# ----------------------------------------------------------
# Observability
# ----------------------------------------------------------

logs: ## Follow logs for all running services
	$(COMPOSE) logs -f

logs-rag: ## Follow RAG server logs only
	$(COMPOSE) logs -f rag

logs-mayan: ## Follow Mayan app logs only
	$(COMPOSE) logs -f app

ps: ## Show running containers and their status
	$(COMPOSE) ps -a

health: ## Quick health check of key services
	@echo "=== Mayan EDMS ==="
	@curl -sf http://localhost:80/ -o /dev/null && echo "  OK (port 80)" || echo "  UNREACHABLE"
	@echo "=== RAG API ==="
	@curl -sf http://localhost:8001/health && echo "" || echo "  UNREACHABLE"
	@echo "=== Qdrant ==="
	@curl -sf http://localhost:6333/healthz && echo "" || echo "  UNREACHABLE"
	@echo "=== Langfuse ==="
	@curl -sf http://localhost:3111/api/public/health -o /dev/null && echo "  OK (port 3111)" || echo "  UNREACHABLE"

# ----------------------------------------------------------
# Build
# ----------------------------------------------------------

build: ## Rebuild the RAG image without starting
	$(COMPOSE) build rag

build-no-cache: ## Rebuild the RAG image from scratch
	$(COMPOSE) build --no-cache rag

# ----------------------------------------------------------
# Shell access
# ----------------------------------------------------------

rag-shell: ## Open a shell in the RAG container
	docker exec -it rag /bin/bash

mayan-shell: ## Open a shell in the Mayan container
	docker exec -it $$(docker compose ps -q app) /bin/bash

# ----------------------------------------------------------
# RAG operations
# ----------------------------------------------------------

index: ## Run the document indexing pipeline
	docker exec -it rag python -m scripts.index_documents

query: ## Launch the interactive query CLI
	docker exec -it rag python -m scripts.query_app

# ----------------------------------------------------------
# Database operations
# ----------------------------------------------------------

backup-db: ## Backup Mayan PostgreSQL to ./backups/
	@mkdir -p backups
	docker exec $$(docker compose ps -q postgresql) \
		pg_dump -U mayan -F c mayan > backups/mayan-$$(date +%Y%m%d_%H%M%S).dump
	@echo "Backup saved to backups/"

# ----------------------------------------------------------
# Cleanup helpers
# ----------------------------------------------------------

prune: ## Remove dangling images and build cache
	docker system prune -f
	docker builder prune -f
