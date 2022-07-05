# noxfile.py
import nox

@nox.session(python=["3.9"])
def tests(session):
    args = session.posargs or ["--cov"]
    session.run_always("poetry", "install", external=True)
    session.install("pytest", "pytest-cov", "coverage[toml]", "pytest-mock")
    session.run("pytest", "--cov")