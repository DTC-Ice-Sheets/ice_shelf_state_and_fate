FROM python:3.13-slim
WORKDIR /app
ENV PORT=80

# Install apt dependencies
RUN apt-get update && apt-get install -y git curl

# Install and configure Poetry
RUN curl -sSL https://install.python-poetry.org | python -
ENV PATH="/root/.local/bin:$PATH"
ENV POETRY_VIRTUALENVS_IN_PROJECT=true

# Install Python dependencies
COPY pyproject.toml poetry.lock /app/
# GitHub token is required for private repositories
RUN --mount=type=secret,id=github_token \
    mkdir -p /root && \
    echo "machine github.com login git password $(cat /run/secrets/github_token)" > /root/.netrc && \
    chmod 600 /root/.netrc
RUN poetry install --no-root --with dashboard
ENV PATH="/app/.venv/bin:$PATH"

# Copy application code and install app
COPY . /app/
RUN poetry install --only-root

# Remove GitHub token after install
RUN rm /root/.netrc

ENTRYPOINT ["poetry", "run", "gunicorn", "--bind", "0.0.0.0:$PORT", "use_case_3.dashboard.app:app"]
