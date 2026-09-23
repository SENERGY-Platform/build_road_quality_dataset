FROM python:3.10

RUN pip install --no-cache-dir uv==0.11.16

WORKDIR /usr/src/app

COPY pyproject.toml uv.lock ./
RUN uv sync --locked

COPY src ./src

ENV PYTHONUNBUFFERED=1

CMD ["/app/.venv/bin/python", "-m", "src.experiments.run_all_optimisations"]
