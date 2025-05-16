FROM node:20-alpine AS frontend-dev

WORKDIR /app

# Copy only package files to cache dependencies
COPY webapp/package*.json ./
RUN npm ci

# Start with base Python image
FROM python:3.10-slim

WORKDIR /app

# Install Node.js for the Next.js frontend
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies for the data service
COPY webapp/data_service/requirements.txt ./webapp/data_service/
# Install toml and other missing dependencies
RUN pip install --no-cache-dir -r webapp/data_service/requirements.txt

# Copy the data service code
COPY webapp/data_service ./webapp/data_service/

# Copy src directory (excluding legacy directory) for supporting Python code
COPY src ./src/
COPY configs ./configs/
COPY scripts ./scripts/
COPY third_party ./third_party/

# Set up the React app (instead of building it)
WORKDIR /app/webapp
COPY --from=frontend-dev /app/node_modules ./node_modules
COPY webapp/package*.json ./
COPY webapp/public ./public
COPY webapp/src ./src
COPY webapp/tsconfig.json ./
COPY webapp/*.mjs ./
COPY webapp/*.ts ./
COPY webapp/*.d.ts ./

# Set environment variables
ENV NEXT_TELEMETRY_DISABLED=1
ENV NODE_ENV=development
ENV NEXT_PUBLIC_API_URL=http://localhost:8000

# Expose ports for both services
EXPOSE 3000 8000

# Create entrypoint script to start both services in development mode
RUN echo '#!/bin/bash \n\
cd /app/webapp/data_service && uvicorn main:app --host 0.0.0.0 --port 8000 & \n\
cd /app/webapp && npm run dev \n\
wait' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"] 