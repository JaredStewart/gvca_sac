.PHONY: help up down logs build rebuild reset prune prune-all status

.DEFAULT_GOAL := help

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

up: ## Start all services
	docker compose up -d

down: ## Stop all services
	docker compose down

logs: ## View logs (follow mode)
	docker compose logs -f

build: ## Build all images
	docker compose build

rebuild: ## Rebuild and restart all services (auto-prunes old images)
	docker compose build && docker compose up -d && docker image prune -f

reset: ## Reset database (removes volumes)
	docker compose down -v && docker compose up -d

prune: ## Remove dangling images (safe)
	docker image prune -f

prune-all: ## Remove dangling images and build cache
	docker image prune -f
	docker builder prune -f

status: ## Show Docker disk usage and dangling images
	@echo "=== Dangling Images ==="
	@docker images -f "dangling=true"
	@echo ""
	@echo "=== Disk Usage ==="
	@docker system df
