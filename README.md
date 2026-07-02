# RedRob Frontend

This branch contains only the RedRob React frontend in a single app folder:

```text
REDROB/
```

The old Python backend files and duplicate nested frontend folders were removed to keep the UI branch clean and easy to run.

## Run Locally

1. Open this repository in VS Code.
2. Open a terminal.
3. Move into the frontend app:

```powershell
cd REDROB
```

4. Install dependencies:

```powershell
npm install
```

5. Start the development server:

```powershell
npm run dev
```

6. Open the local URL shown by Vite, usually:

```text
http://127.0.0.1:5173/
```

## Build

```powershell
cd REDROB
npm run build
```

The production build is written to `REDROB/dist/`, which is intentionally ignored by Git.

## Environment

The app runs with mock data by default. To connect live services, copy `REDROB/.env.example` to `REDROB/.env` and fill in the values.

```text
VITE_USE_MOCK_API=true
VITE_API_URL=http://localhost:8000
VITE_SUPABASE_URL=replace-with-supabase-url
VITE_SUPABASE_ANON_KEY=replace-with-supabase-anon-key
```

## Kept Files

The UI branch keeps:

- `REDROB/src/`
- `REDROB/package.json`
- `REDROB/package-lock.json`
- `REDROB/index.html`
- `REDROB/vite.config.js`
- `REDROB/tailwind.config.js`
- `REDROB/docs/`
- frontend documentation files

The UI branch removes:

- old Python backend files
- old tests for the backend/ranker
- duplicate nested `REDROB/REDROB/` frontend copies
- generated folders such as `node_modules/` and `dist/`
