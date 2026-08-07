FROM python:3.10-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    POETRY_VERSION=1.7.1 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    APP_DIR="/app"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="$POETRY_HOME/bin:$PATH"

WORKDIR $APP_DIR

# Copy dependency files
COPY pyproject.toml poetry.lock* ./

# Install dependencies
RUN poetry install --no-root --without dev

# Copy application code
COPY . .

# Install application
RUN poetry install --without dev

# Create a non-root user
RUN useradd -m -U appuser && \
    chown -R appuser:appuser $APP_DIR

USER appuser

EXPOSE 8000

CMD ["poetry", "run", "uvicorn", "biooncograph.main:app", "--host", "0.0.0.0", "--port", "8000"]
