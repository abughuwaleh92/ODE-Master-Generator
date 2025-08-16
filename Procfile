# Streamlit UI service
web: streamlit run master_generators_app.py --server.port $PORT --server.address 0.0.0.0

# Worker service
worker: rq worker -u $REDIS_URL ode_jobs
