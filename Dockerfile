# One image for the whole pipeline: `docker compose run --rm costmodel <command>`.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    COSTMODEL_HOME=/work \
    MPLBACKEND=Agg

# LightGBM needs the OpenMP runtime, which the slim image does not ship.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so code changes do not reinstall them.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-install-project

COPY src ./src
COPY tests ./tests
COPY configs ./configs
COPY templates ./templates
COPY data/checksums ./data/checksums
RUN uv sync --locked

# Data and reports are written to /work (a volume), never into the image.
RUN useradd --create-home --uid 1000 app && mkdir -p /work && chown app /work
USER app
VOLUME /work

ENTRYPOINT ["costmodel"]
CMD ["--help"]
