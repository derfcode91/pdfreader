# file-data-extraction

Experiment extract data from PDF files using AI (Gemini API)

## Docker setup

**Ubuntu 24.04:** The `docker-compose` from apt is broken with Python 3.12. Install Compose V2:  
`sudo apt install docker-compose-v2`  
Then use `docker compose` (with a space), e.g. `docker compose up --build -d`.

1. **Create env file** (required for Gemini):

   ```bash
   cp .env.example .env
   ```

   Edit `.env` and set `GEMINI_API_KEY` (and optionally `SECRET_KEY` for production).

2. **Build and run** (from this directory):

   - **Docker Compose V2** (plugin, `docker compose`):

     ```bash
     docker compose up --build -d
     ```

   - **Docker Compose V1** (standalone, `docker-compose`):

     ```bash
     docker-compose up --build -d
     ```

   If `--build` gives an error, build then start in two steps:

   ```bash
   docker-compose build
   docker-compose up -d
   ```

   App is at **http://localhost:5001**. To run in foreground (see logs): `docker-compose up --build` or `docker compose up --build`.

3. **Stop**:

   ```bash
   docker-compose down
   ```

   (Use `docker compose down` if you use the V2 plugin.)

