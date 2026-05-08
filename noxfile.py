import nox


@nox.session(reuse_venv=True)
def lint(session: nox.Session) -> None:
    session.run("uv", "sync", "--dev", "--active", external=True)
    session.run("ruff", "check", ".", "--fix", external=True)
    session.run("ruff", "format", ".", external=True)


@nox.session(reuse_venv=True)
def typecheck(session: nox.Session) -> None:
    session.run("uv", "sync", "--dev", "--active", external=True)
    session.run("mypy", "src", external=True)


@nox.session(reuse_venv=True)
def tests(session: nox.Session) -> None:
    session.run("uv", "sync", "--dev", "--active", external=True)
    session.run("pytest", "tests", external=True, success_codes=[0, 5])
