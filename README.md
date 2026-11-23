# MiraiQ

Personal anime recommendation engine powered by your AniList taste. A three-part project:

| Part | Directory | Description |
|------|-----------|-------------|
| **Python package** | `package/` | Standalone `miraiq` CLI + library (sync, train, recommend, …) |
| **Backend** | `backend/` | Flask HTTP API wrapping `miraiq` for a React frontend |
| **Frontend** | `web/` | React TypeScript app (Vite) |

## Install

**Python package (CLI):**
```bash
cd package
pip install -e .
miraiq sync your_username
miraiq recommend your_username
```

**Backend:**
```bash
cd backend
pip install -r requirements.txt
python run.py
```

**Frontend:**
```bash
cd web
bun install
bun run dev
```

Requires Python 3.10+ for the package/backend, Node.js for the frontend.

## License

MIT — see [LICENSE](LICENSE)
