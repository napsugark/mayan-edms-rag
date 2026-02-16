#!make
include config.env

ifneq ($(wildcard config-local.env),)
	include config-local.env
endif

IMAGE_VERSION ?= `cat docker/rootfs/version`
FORGE_DOCKER_IMAGE_NAME ?= $(FORGE_DOCKER_IMAGE_NAME)
DOCKER_IMAGE_DATE_TIME ?= `date -Iseconds  --utc`
DOCKER_IMAGE_LABELS ?= --label org.opencontainers.image.created="$(DOCKER_IMAGE_DATE_TIME)" --label org.opencontainers.image.version=$(IMAGE_VERSION) $(DOCKER_IMAGE_LABELS_EXTRA)
MAYAN_TEST_MEDIA_ROOT ?= /tmp/mayan-test

# Build

forge-buildkitd-config-create:
	@echo "debug = true" > /tmp/buildkitd.toml
	@if [ $(DOCKER_MIRROR) ]; then \
		echo "" >> /tmp/buildkitd.toml; \
		echo '[registry."docker.io"]' >> /tmp/buildkitd.toml; \
		echo '  mirrors = ["$(DOCKER_MIRROR)"]' >> /tmp/buildkitd.toml; \
	fi

forge-build: ## Build a new image locally.
forge-build: forge-buildkitd-config-create
	echo "$(IMAGE_VERSION)"
	docker buildx rm mirrored || true
	docker context rm tls-context || true
	docker context create tls-context
	docker buildx create --bootstrap --config /tmp/buildkitd.toml --driver docker-container --name mirrored --use tls-context
	DOCKER_BUILDKIT=1 docker build --build-arg APT_PROXY=$(APT_PROXY) --build-arg PIP_INDEX_URL=$(PIP_INDEX_URL) --build-arg PIP_TRUSTED_HOST=$(PIP_TRUSTED_HOST) --build-arg HTTP_PROXY=$(HTTP_PROXY) --build-arg HTTPS_PROXY=$(HTTPS_PROXY) --builder mirrored --file forge/Dockerfile $(DOCKER_IMAGE_LABELS) --output type=docker --tag $(FORGE_DOCKER_IMAGE_NAME):$(IMAGE_VERSION) .
	docker buildx stop mirrored

forge-dockerfile-update: ## Update the Dockerfile file from the platform template.
forge-dockerfile-update: config-env-copy
	./manage.py platforms_template forge_dockerfile > forge/Dockerfile

forge-docker-compose-update: ## Update the Docker Compose file from the platform template.
forge-docker-compose-update: config-env-copy
	./manage.py platforms_template forge_docker_compose > forge/docker-compose.yml

# Run

forge-down: ## Create and start the Forge container.
	@docker compose --file forge/docker-compose.yml down

forge-shell: ## Launch a bash instance inside the Forge container.
forge-shell: forge-up
	@docker compose --file forge/docker-compose.yml exec --env TERM=$(TERM) --interactive --tty forge bash

forge-up: ## Create and start the Forge container.
	@docker compose --file forge/docker-compose.yml up --detach

forge-update: ## Update all Forge Docker files.
forge-update: forge-docker-compose-update forge-dockerfile-update
