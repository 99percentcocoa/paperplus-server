import multiprocessing

# Server socket
bind = "127.0.0.1:8001"  # Nginx will reverse proxy to this
backlog = 2048

# Worker processes
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "sync"
worker_connections = 1000
timeout = 120  # Increased for image processing
keepalive = 5

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