# Use Python 3.12.5 as base image
FROM python:3.12.5-slim

# Set working directory in container
WORKDIR /app

# Install PostgreSQL and required packages
RUN apt-get update && \
    apt-get install -y postgresql postgresql-contrib libpq-dev gcc && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir pandas yfinance numpy psycopg2-binary

# Copy your files
COPY main.py .
COPY db/schema.sql /docker-entrypoint-initdb.d/

# Create a startup script
RUN echo '#!/bin/bash\n\
service postgresql start\n\
su - postgres -c "psql -f /docker-entrypoint-initdb.d/schema.sql"\n\
python main.py' > /start.sh && \
chmod +x /start.sh

# Run the startup script
CMD ["/start.sh"]
