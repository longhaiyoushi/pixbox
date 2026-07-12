# mypy: ignore-errors

import nox

PYTHON_VERSIONS = ['3.12', '3.13', '3.14']

nox.options.default_venv_backend = 'uv'
nox.options.sessions = ['ruff_check', 'ruff_format', 'mypy_check']


@nox.session(python=PYTHON_VERSIONS)
def ruff_check(session: nox.Session):
    session.install('ruff')
    session.run('ruff', 'check', '.')


@nox.session(python=PYTHON_VERSIONS)
def ruff_format(session: nox.Session):
    session.install('ruff')
    session.run('ruff', 'format', '.', '--check')


@nox.session(python=PYTHON_VERSIONS)
def mypy_check(session: nox.Session):
    session.run('uv', 'sync', '--frozen')
    session.install('mypy')
    session.run('mypy', '.', '--strict')
