#!make
include config.env

ifneq ($(wildcard config-local.env),)
	include config-local.env
endif

ifndef MODULE
override MODULE = --mayan-apps
endif

ifdef TAG
override ARGUMENT_TAG = --tag=$(TAG)
endif

ifndef SKIPMIGRATIONS
override SKIPMIGRATIONS = --skip-migrations
endif

ifndef SETTINGS
override SETTINGS = mayan.settings.testing.development
endif

ifeq ($(origin APT_PROXY), undefined)
	ifneq ($(origin APT_PROXY_IP), undefined)
		APT_PROXY = "$(APT_PROXY_IP):3142"
	endif
endif

ifeq ($(origin PIP_INDEX_URL), undefined)
	ifneq ($(origin PIP_PROXY_IP), undefined)
		PIP_INDEX_URL = "http://$(PIP_PROXY_IP):3141/root/pypi/+simple/"
	endif
endif

ifeq ($(origin PIP_TRUSTED_HOST), undefined)
	PIP_TRUSTED_HOST = "$(PIP_PROXY_IP)"
endif

COMMAND_SENTRY = \
	if [ $(SENTRY_DSN) ]; then \
	export MAYAN_PLATFORMS_CLIENT_BACKEND_ENABLED='["mayan.apps.platforms_sentry.client_backends.ClientBackendSentry"]'; \
	export MAYAN_PLATFORMS_CLIENT_BACKEND_ARGUMENTS='{"mayan.apps.platforms_sentry.client_backends.ClientBackendSentry":{"dsn":"$(SENTRY_DSN)","environment":"development"}}'; \
	fi

COMMAND_TEST = ./manage.py test $(MODULE) --settings=$(SETTINGS) $(SKIPMIGRATIONS) $(DEBUG) $(ARGUMENTS) $(ARGUMENT_TAG)
COMMAND_TEST_MIGRATIONS = ./manage.py test $(MODULE) --no-exclude --settings=$(SETTINGS) --tag=migration_test $(DEBUG) $(ARGUMENTS)

CONTAINER_NAME_TEST_ELASTIC = mayan-test-elastic
CONTAINER_NAME_TEST_MYSQL = mayan-test-mysql
CONTAINER_NAME_TEST_ORACLE = mayan-test-oracle
CONTAINER_NAME_TEST_POSTGRESQL = mayan-test-postgresql
CONTAINER_NAME_TEST_REDIS = mayan-test-redis

.PHONY: clean clean-pyc clean-build test

help:
	@echo "Usage: make <target>\n"
	@awk 'BEGIN {FS = ":.*##"} /^[0-9a-zA-Z_-]+:.*?## / { printf "  * %-40s -%s\n", $$1, $$2 }' $(MAKEFILE_LIST)|sort

# Cleaning

clean: ## Remove Python and build artifacts.
clean: clean-build clean-pyc

clean-build: ## Remove build artifacts.
	rm --force --recursive build/
	rm --force --recursive dist/
	rm --force --recursive *.egg-info

clean-pyc: ## Remove Python artifacts.
	find . -name '*.pyc' -exec rm --force {} +
	find . -name '*.pyo' -exec rm --force {} +
	find . -name '*~' -exec rm --force {} +
	find . -name '__pycache__' -exec rm --force --recursive {} +

# Testing

_test-command:
	$(COMMAND_TEST)

test: ## MODULE=<python module name> - Run tests for a single app, module or test class.
test: clean-pyc _test-command

test-debug: ## MODULE=<python module name> - Run tests for a single app, module or test class, in debug mode.
test-debug: DEBUG=--debug-mode
test-debug: clean-pyc _test-command

test-all: ## Run all tests.
test-all: clean-pyc _test-command

test-all-debug: ## Run all tests in debug mode.
test-all-debug: DEBUG=--debug-mode
test-all-debug: clean-pyc _test-command

test-with-mysql: ## MODULE=<python module name> - Run tests for a single app, module or test class against a MySQL database container.
test-with-mysql:
	export MAYAN_DATABASES="{'default':{'ENGINE':'django.db.backends.mysql','NAME':'$(DEFAULT_DATABASE_NAME)','PASSWORD':'$(DEFAULT_DATABASE_PASSWORD)','USER':'$(DEFAULT_DATABASE_USER)','HOST':'127.0.0.1'}}"; \
	$(COMMAND_TEST)

test-all-with-mysql: ## Run all tests against a MySQL database container.
test-all-with-mysql:
	export MAYAN_DATABASES="{'default':{'ENGINE':'django.db.backends.mysql','NAME':'$(DEFAULT_DATABASE_NAME)','PASSWORD':'$(DEFAULT_DATABASE_PASSWORD)','USER':'$(DEFAULT_DATABASE_USER)','HOST':'127.0.0.1'}}"; \
	$(COMMAND_TEST)

test-with-oracle: ## MODULE=<python module name> - Run tests for a single app, module or test class against an Oracle database container.
test-with-oracle:
	export MAYAN_DATABASES="{'default':{'ENGINE':'django.db.backends.oracle','NAME':'$(DEFAULT_DATABASE_NAME)','PASSWORD':'$(DEFAULT_DATABASE_PASSWORD)','USER':'$(DEFAULT_DATABASE_USER)','HOST':'127.0.0.1'}}"; \
	$(COMMAND_TEST)

test-all-with-oracle: ## Run all tests against an Oracle database container.
test-all-with-oracle:
	export MAYAN_DATABASES="{'default':{'ENGINE':'django.db.backends.oracle','NAME':'$(DEFAULT_DATABASE_NAME)','PASSWORD':'$(DEFAULT_DATABASE_PASSWORD)','USER':'$(DEFAULT_DATABASE_USER)','HOST':'127.0.0.1'}}"; \
	$(COMMAND_TEST)

test-with-postgresql: ## MODULE=<python module name> - Run tests for a single app, module or test class against a PostgreSQL database container.
test-with-postgresql:
	export MAYAN_DATABASES="{'default':{'ENGINE':'django.db.backends.postgresql','NAME':'$(DEFAULT_DATABASE_NAME)','PASSWORD':'$(DEFAULT_DATABASE_PASSWORD)','USER':'$(DEFAULT_DATABASE_USER)','HOST':'127.0.0.1'}}"; \
	$(COMMAND_TEST)

test-all-with-postgresql: ## Run all tests against a PostgreSQL database container.
test-all-with-postgresql:
	export MAYAN_DATABASES="{'default':{'ENGINE':'django.db.backends.postgresql','NAME':'$(DEFAULT_DATABASE_NAME)','PASSWORD':'$(DEFAULT_DATABASE_PASSWORD)','USER':'$(DEFAULT_DATABASE_USER)','HOST':'127.0.0.1'}}"; \
	$(COMMAND_TEST)

# Migrations

_test-migrations:
_test-migrations: ARGUMENTS=--no-exclude --tag=migration_test
_test-migrations: SKIPMIGRATIONS=
_test-migrations: clean-pyc _test-command

test-migrations: ## Run specific migration tests.
test-migrations: _test-migrations

test-migrations-all: ## Run all migration tests.
test-migrations-all: _test-migrations

test-migrations-all-with-mysql: ## Run all migration tests against a MySQL database container.
test-migrations-all-with-mysql:
	export MAYAN_DATABASES="{'default':{'ENGINE':'django.db.backends.mysql','NAME':'$(DEFAULT_DATABASE_NAME)','PASSWORD':'$(DEFAULT_DATABASE_PASSWORD)','USER':'$(DEFAULT_DATABASE_USER)','HOST':'127.0.0.1'}}"; \
	$(COMMAND_TEST_MIGRATIONS)

test-migrations-all-with-oracle: ## Run all migration tests against an Oracle database container.
test-migrations-all-with-oracle:
	export MAYAN_DATABASES="{'default':{'ENGINE':'django.db.backends.oracle','NAME':'$(DEFAULT_DATABASE_NAME)','PASSWORD':'$(DEFAULT_DATABASE_PASSWORD)','USER':'$(DEFAULT_DATABASE_USER)','HOST':'127.0.0.1'}}"; \
	$(COMMAND_TEST_MIGRATIONS)

test-migrations-all-with-postgresql: ## Run all migration tests against a PostgreSQL database container.
test-migrations-all-with-postgresql:
	export MAYAN_DATABASES="{'default':{'ENGINE':'django.db.backends.postgresql','NAME':'$(DEFAULT_DATABASE_NAME)','PASSWORD':'$(DEFAULT_DATABASE_PASSWORD)','USER':'$(DEFAULT_DATABASE_USER)','HOST':'127.0.0.1'}}"; \
	$(COMMAND_TEST_MIGRATIONS)

test-migrations-with-postgresql: ## MODULE=<python module name> - Run migration tests for a single app, module or test class against a PostgreSQL database container.
test-migrations-with-postgresql:
	export MAYAN_DATABASES="{'default':{'ENGINE':'django.db.backends.postgresql','NAME':'$(DEFAULT_DATABASE_NAME)','PASSWORD':'$(DEFAULT_DATABASE_PASSWORD)','USER':'$(DEFAULT_DATABASE_USER)','HOST':'127.0.0.1'}}"; \
	$(COMMAND_TEST_MIGRATIONS)

# Coverage

coverage-run: ## Run all tests and measure code execution.
coverage-run: clean-pyc
	coverage run $(COMMAND_TEST)

coverage-html: ## Create the coverage HTML report. Run execute coverage-run first.
coverage-html:
	coverage html

# Documentation

docs-app-documentation-generate: ## Collect app documentation into the docs folder
docs-app-documentation-generate:
	./contrib/scripts/app_documentation_generator.py

docs-html: ## Run the html documentation target. Use optional FILENAMES to specific which files to build.
docs-html: docs-app-documentation-generate
	cd docs;make html

docs-spellcheck: ## Spellcheck the documentation.
	cd docs;sphinx-build -b spelling -d _build/ . _build/spelling

# Translations

translations-source-clear: ## Clear the msgstr of the source file
	@sed -i -E  's/msgstr ".+"/msgstr ""/g' `grep -E 'msgstr ".+"' mayan/apps/*/locale/en/*/django.po | cut -d: -f 1` > /dev/null 2>&1  || true

translations-source-fuzzy-remove: ## Remove fuzzy makers
	sed -i  '/#, fuzzy/d' mayan/apps/*/locale/*/LC_MESSAGES/django.po

translations-transifex-check: ## Check that all app have a Transifex entry
	contrib/scripts/translations_helper.py transifex_missing_apps

translations-transifex-generate: ## Check that all app have a Transifex entry
	contrib/scripts/translations_helper.py transifex_generate_config > ./.tx/config

translations-django-make: ## Refresh all translation files.
	contrib/scripts/translations_helper.py django_make

translations-django-make-javascript: ## Refresh all JavaScript translation files.
	contrib/scripts/translations_helper.py django_make_javascript

translations-django-compile: ## Compile all translation files.
	contrib/scripts/translations_helper.py django_compile

translations-transifex-push: ## Upload all translation files to Transifex.
	tx push -s

translations-transifex-pull: ## Download all translation files from Transifex.
	tx pull -f

translations-all: ## Execute all translations targets.
translations-all: translations-source-clear translations-source-fuzzy-remove translations-transifex-generate translations-django-make translations-django-make-javascript translations-transifex-push translations-transifex-pull translations-django-compile

# Releases

requirements-generate: ## Generate all requirements files from the project dependency declarations.
	@./manage.py dependencies_generate_requirements build --settings=mayan.settings.development > requirements/build.txt
	@./manage.py dependencies_generate_requirements development --settings=mayan.settings.development > requirements/development.txt
	@./manage.py dependencies_generate_requirements documentation --settings=mayan.settings.development > requirements/documentation.txt
	@./manage.py dependencies_generate_requirements documentation_override --settings=mayan.settings.development > requirements/documentation_override.txt
	@./manage.py dependencies_generate_requirements testing --settings=mayan.settings.testing > requirements/testing-base.txt
	@./manage.py dependencies_generate_requirements production --exclude=django > requirements/base.txt
	@./manage.py dependencies_generate_requirements production --only=django > requirements/common.txt

version-increase: ## Increase the version number of the entire project's files.
	@VERSION_BASE=`grep "__version__ =" mayan/__init__.py| cut -d\' -f 2|./contrib/scripts/increase_version.py - $(PART)`; \
	VERSION=`mayan/apps/dependencies/versions.py $${VERSION_BASE} upstream`; \
	VERSION_PYTHON=`if [ -z "${LOCAL_VERSION}" ]; then echo "$${VERSION}"; else echo "$${VERSION}+${LOCAL_VERSION}"; fi`; \
	sed -i -e "s/__version__ = '[0-9\.a-zA-Z\+]*'/__version__ = '$$VERSION_PYTHON'/g" mayan/__init__.py; \
	make versions-update; \
	make python-setup-generate

versions-update: ## Update the version number of the entire project's files.
versions-update: config-env-copy
	@VERSION_BASE=`grep "__version__ =" mayan/__init__.py| cut -d\' -f 2`; \
	VERSION=`mayan/apps/dependencies/versions.py $${VERSION_BASE} upstream;` \
	VERSION_PYTHON=`if [ -z "${LOCAL_VERSION}" ]; then echo "$${VERSION}"; else echo "$${VERSION}+${LOCAL_VERSION}"; fi`; \
	VERSION_DOCKER=`if [ -z "${LOCAL_VERSION}" ]; then echo "$${VERSION}"; else echo "$${VERSION}-${LOCAL_VERSION}"; fi`; \
	BUILD=`echo $$VERSION_PYTHON|awk '{split($$VERSION_PYTHON,a,"."); printf("0x%02d%02d%02d\n", a[1],a[2], a[3])}'`; \
	sed -i -e "s/__build__ = 0x[0-9]*/__build__ = $${BUILD}/g" mayan/__init__.py; \
	sed -i -e "s/__version__ = '[0-9\.a-zA-Z\+]*'/__version__ = '$$VERSION_PYTHON'/g" mayan/__init__.py; \
	echo $$VERSION_DOCKER > docker/rootfs/version

# Dev server

manage: ## Run a command with the development settings.
	./manage.py $(filter-out $@,$(MAKECMDGOALS)) --settings=mayan.settings.development

manage-with-mysql: ## Run the development server using a Docker PostgreSQL container.
	@export MAYAN_DATABASES="{'default':{'ENGINE':'django.db.backends.mysql','NAME':'$(DEFAULT_DATABASE_NAME)','PASSWORD':'$(DEFAULT_DATABASE_PASSWORD)','USER':'$(DEFAULT_DATABASE_USER)','HOST':'127.0.0.1'}}"; \
	./manage.py $(filter-out $@,$(MAKECMDGOALS)) --settings=mayan.settings.development

manage-with-oracle: ## Run the development server using a Docker Oracle container.
	@export MAYAN_DATABASES="{'default':{'ENGINE':'django.db.backends.oracle','NAME':'$(DEFAULT_DATABASE_NAME)','PASSWORD':'$(DEFAULT_DATABASE_PASSWORD)','USER':'$(DEFAULT_DATABASE_USER)','HOST':'127.0.0.1'}}"; \
	./manage.py $(filter-out $@,$(MAKECMDGOALS)) --settings=mayan.settings.development

manage-with-postgresql: ## Run the development server using a Docker PostgreSQL container.
	@export MAYAN_DATABASES="{'default':{'ENGINE':'django.db.backends.postgresql','NAME':'$(DEFAULT_DATABASE_NAME)','PASSWORD':'$(DEFAULT_DATABASE_PASSWORD)','USER':'$(DEFAULT_DATABASE_USER)','HOST':'127.0.0.1'}}"; \
	./manage.py $(filter-out $@,$(MAKECMDGOALS)) --settings=mayan.settings.development

runserver: ## Run the development server.
	$(COMMAND_SENTRY); ./manage.py runserver --settings=mayan.settings.development $(ADDRPORT)

runserver-plus: ## Run the Django extension's development server.
	$(COMMAND_SENTRY); ./manage.py runserver_plus --settings=mayan.settings.development $(ADDRPORT)

shell-plus: ## Run the shell_plus command.
	./manage.py shell_plus --settings=mayan.settings.development

# Other

find-gitignores: ## Find stray .gitignore files.
	@export FIND_GITIGNORES=`find -name '.gitignore'| wc -l`; \
	if [ $${FIND_GITIGNORES} -gt 1 ] ;then echo "More than one .gitignore found: $$FIND_GITIGNORES"; fi

check-readme: ## Checks validity of the README.rst file for PyPI publication.
	python setup.py check --restructuredtext --strict

check-missing-migrations: ## Make sure all models have proper migrations.
	./manage.py makemigrations --dry-run --noinput --check

check-missing-inits: ## Find missing __init__.py files from modules.
	@contrib/scripts/find_missing_inits.py

config-env-copy: ## Copy and convert `config.env` to `settings.literals.py`.
	@contrib/scripts/copy_config_env.py > mayan/settings/literals.py

# Development environment

dev-setup: ## Bootstrap a virtualenv by install all dependencies to start developing.
dev-setup: dev-setup-os-ubuntu dev-setup-python

dev-setup-os-ubuntu:  ## Install the operating system packages needed for development.
	sudo apt-get install --yes clamav exiftool gcc gettext gnupg1 graphviz libcairo2 libffi-dev libfuse2 libjpeg-dev libldap2-dev libpng-dev libsasl2-dev poppler-utils python3-dev sane-utils tesseract-ocr-deu


-include docker/Makefile
-include forge/Makefile
-include gitlab-ci/Makefile
-include python/Makefile
