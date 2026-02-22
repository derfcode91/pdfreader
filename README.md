# file-data-extraction

Experiment extract data from PDF files using AI (Gemini API)

## Docker setup

1. **Create env file** (required for Gemini):

   ```bash
   cp .env.example .env
   ```

   Edit `.env` and set `GEMINI_API_KEY` (and optionally `SECRET_KEY` for production).

2. **Build and run** (from this directory):

   ```bash
   docker compose up --build -d
   ```

   App is at **http://localhost:5001**. To run in foreground (see logs): `docker compose up --build`.

3. **Stop**:

   ```bash
   docker compose down
   ```

Volumes: `./uploads`, `./data`, and `./templates` are mounted so uploads and DB persist and you can edit templates without rebuilding.

**Without Docker Compose plugin** (if `docker compose` is not available):

```bash
sudo ./docker-build-run.sh
```

Then open http://localhost:5001. To stop: `sudo docker rm -f pdfreader-web-2`.
