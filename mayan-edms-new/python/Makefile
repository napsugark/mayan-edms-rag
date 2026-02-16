#!make
include config.env

ifneq ($(wildcard config-local.env),)
        include config-local.env
endif

python-setup-generate: ## Create and update the setup.py file.
python-setup-generate: requirements-generate
	@./contrib/scripts/generate_setup.py
	@echo "Complete."

python-release: ## Package (sdist and wheel) and upload a release.
python-release: clean python-wheel-production
	twine upload dist/* -r pypi

python-sdist-production: ## Build the production source distribution package.
python-sdist-production: clean
	cp MANIFEST-PRODUCTION.in MANIFEST.in
	python setup.py sdist
	ls -l dist

python-sdist-testing: ## Build the testing source distribution package.
python-sdist-testing: clean
	cp MANIFEST-TESTING.in MANIFEST.in
	python setup.py sdist
	ls -l dist

python-wheel-production: ## Build the production wheel distribution package.
python-wheel-production: clean python-sdist-production
	pip wheel --no-index --no-deps --wheel-dir dist dist/*.tar.gz
	ls -l dist

python-wheel-testing: ## Build the testing wheel distribution package.
python-wheel-testing: clean python-sdist-testing
	pip wheel --no-index --no-deps --wheel-dir dist dist/*.tar.gz
	ls -l dist

dev-setup-python: ## Install the Python libraries needed for development.
	pip install --requirement requirements.txt --requirement requirements/development.txt --requirement requirements/testing-base.txt --requirement requirements/documentation.txt --requirement requirements/build.txt

dev-setup-python-mysql: ## Install the Python MySQL library.
	@pip install mysqlclient==$(PYTHON_MYSQL_VERSION)

dev-setup-python-oracle: ## Install the Python Oracle library.
	@pip install cx_Oracle==$(PYTHON_ORACLE_VERSION)

dev-setup-python-postgresql: ## Install the Python PostgreSQL library.
	@pip install psycopg==$(PYTHON_PSYCOPG_VERSION)

dev-setup-python-redis:
	@pip install redis==$(PYTHON_REDIS_VERSION)
