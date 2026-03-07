FROM node:20-alpine

WORKDIR /app

# Optionally inject corporate CA cert for builds behind TLS-intercepting proxies
RUN --mount=type=secret,id=ca_cert,target=/tmp/ca-cert.pem \
    if [ -f /tmp/ca-cert.pem ]; then \
      mkdir -p /usr/local/share/ca-certificates && \
      cp /tmp/ca-cert.pem /usr/local/share/ca-certificates/custom-ca.crt && \
      cat /tmp/ca-cert.pem >> /etc/ssl/certs/ca-certificates.crt; \
    fi

# Install dependencies
COPY package.json package-lock.json* ./
RUN if [ -f /usr/local/share/ca-certificates/custom-ca.crt ]; then \
      NODE_EXTRA_CA_CERTS=/usr/local/share/ca-certificates/custom-ca.crt npm install; \
    else \
      npm install; \
    fi

# Copy application code
COPY . .

# Expose port
EXPOSE 5173

# Run development server
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
