import multiprocessing

# Server socket
bind = "127.0.0.1:8001"  # Nginx will reverse proxy to this
backlog = 2048

# Keep worker count conservative for ML-heavy workloads.
# The earlier crash showed import-time failures under a large worker fan-out.
workers = max(2, min(4, (multiprocessing.cpu_count() // 2) or 2))
worker_class = "sync"
worker_connections = 1000
timeout = 180
keepalive = 5
max_requests = 1000
max_requests_jitter = 100
graceful_timeout = 30
preload_app = False

# Logging
accesslog = "logs/gunicorn-access.log"
errorlog = "logs/gunicorn-error.log"
loglevel = "info"

# Process naming
proc_name = "paperplus-server"

# Server mechanics
daemon = False
pidfile = "logs/gunicorn.pid"
umask = 0
user = None
group = None
tmp_upload_dir = None