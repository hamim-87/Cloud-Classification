# Eyes on Cloud — Docker Deployment

A containerized satellite imagery selector built with CesiumJS and FastAPI.

## 🚀 Quick Start

### Prerequisites
- Docker
- Docker Compose

### Run with Docker Compose

```bash
docker-compose up -d
```

This will start:
- **Frontend**: http://localhost:3000 (Nginx serving CesiumJS app)
- **Backend API**: http://localhost:8000 (FastAPI)

### View Logs

```bash
docker-compose logs -f frontend
docker-compose logs -f backend
```

### Stop All Services

```bash
docker-compose down
```

### Rebuild Images

```bash
docker-compose up --build -d
```

## 📁 Directory Structure

```
eyes-on-earth/
├── docker-compose.yml          # Multi-container orchestration
├── .dockerignore                # Files to exclude from Docker builds
├── index.html                   # Main frontend page
├── app.js                        # CesiumJS application (no comments)
├── style.css                     # Frontend styling
├── frontend/
│   ├── Dockerfile              # Nginx container for static assets
│   └── nginx.conf              # Nginx configuration
├── backend/
│   ├── Dockerfile              # Python/FastAPI container
│   ├── main.py                 # FastAPI application
│   └── requirements.txt         # Python dependencies
├── data/                        # Mounted volume for saved satellite images
└── README.md                    # This file
```

## 🐳 Docker Container Details

### Frontend (Nginx)

- **Image**: `nginx:alpine`
- **Port**: 3000
- **Serves**: Static HTML/CSS/JS
- **Features**:
  - Gzip compression
  - Cache headers for static assets
  - SPA routing (404 → index.html)

### Backend (FastAPI)

- **Image**: `python:3.12-slim`
- **Port**: 8000
- **Features**:
  - CORS enabled (all origins)
  - Image upload endpoint: `POST /save-image`
  - Health check: `GET /`
  - Auto-restart on failure

## 🔧 API Endpoints

### Health Check
```bash
curl http://localhost:8000/
```

### Save Image
```bash
curl -X POST http://localhost:8000/save-image \
  -F "image=@satellite.png" \
  -F "filename=satellite_2026-04-01.png"
```

### List Saved Images
```bash
curl http://localhost:8000/images
```

## 💾 Data Persistence

Saved satellite images are stored in the `data/` folder, which is mounted as a Docker volume. Images persist even after containers stop.

## 🔗 Inter-Container Communication

Both services run on the same Docker network (`eyes-network`). Backend health check runs on `http://localhost:8000` (container's perspective).

## 📝 Environment Variables

Currently none required. Backend URL auto-detects based on browser's hostname.

## 🛠️ Development

### Build Individual Images

**Frontend**:
```bash
docker build -f frontend/Dockerfile -t eyes-on-earth-frontend .
docker run -p 3000:80 eyes-on-earth-frontend
```

**Backend**:
```bash
docker build -f backend/Dockerfile -t eyes-on-earth-backend .
docker run -p 8000:8000 eyes-on-earth-backend
```

### Local Development (Without Docker)

**Frontend**:
```bash
python -m http.server 3000
```

**Backend**:
```bash
cd backend
pip install -r requirements.txt
python main.py
```

## 🐛 Troubleshooting

### "Connection refused" to backend

- Ensure `docker-compose up -d` completed successfully
- Check backend health: `curl http://localhost:8000/`
- View backend logs: `docker-compose logs backend`

### Port already in use

- Frontend: `docker-compose down && docker-compose up -d`
- Or change ports in `docker-compose.yml`

### Data folder not created

- Backend container creates `/app/data` automatically
- On host, appears as `./data/`
- Ensure write permissions on host

## 📦 Image Sizes

- Frontend (Nginx): ~40 MB
- Backend (Python 3.12 slim): ~150 MB
- Total: ~190 MB

## 🚢 Production Notes

For production deployment:
- Use environment variables for sensitive config
- Add SSL/TLS termination (reverse proxy)
- Use a dedicated database instead of file storage
- Implement rate limiting and API authentication
- Use multi-stage Docker builds for smaller images
- Add health checks with proper timeouts
- Configure resource limits (CPU, memory)

---

**Version**: 1.0.0  
**Last Updated**: April 1, 2026
