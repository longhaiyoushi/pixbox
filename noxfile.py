# mypy: ignore-errors

import nox

PYTHON_VERSIONS = ['3.12', '3.13', '3.14']

nox.options.default_venv_backend = 'uv'
nox.options.sessions = ['lint', 'format', 'types', 'tests']


@nox.session(python=PYTHON_VERSIONS)
def lint(session: nox.Session):
    session.install('ruff')
    session.run('ruff', 'check', '.')


@nox.session(python=PYTHON_VERSIONS)
def format(session: nox.Session):
    session.install('ruff')
    session.run('ruff', 'format', '.', '--check')


@nox.session(python=PYTHON_VERSIONS)
def types(session: nox.Session):
    session.run('uv', 'sync', '--frozen')
    session.install('mypy')
    session.run('mypy', '.', '--strict')


@nox.session(python=PYTHON_VERSIONS)
def tests(session: nox.Session):
    session.run('uv', 'sync', '--frozen')
    session.install('pytest', 'pytest-cov')
    session.run('pytest', 'tests', '--cov=src', '--cov-report=term-missing')
