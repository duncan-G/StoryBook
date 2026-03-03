FROM postgres:18

# Update package list and install pg_cron and pgvector
RUN apt-get update && \
    apt-get install -y postgresql-18-cron postgresql-18-pgvector && \
    rm -rf /var/lib/apt/lists/*