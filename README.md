# file-data-extraction

Extract data from PDF files using AI (Gemini API).

## Setup

1. **Create `.env`** (required for Gemini):

   ```bash
   cp .env.example .env
   ```

   Edit `.env` and set your `GEMINI_API_KEY`.

2. **Build and run** (from this directory):

   ```bash
   docker compose up --build -d
   ```

   Open **http://localhost:5001** in your browser.

3. **Stop** when done:

   ```bash
   docker compose down
   ```

---

**Other commands:**  
View logs: `docker compose logs -f`  
Run in foreground (see logs in terminal): `docker compose up --build`
