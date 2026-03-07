FROM alpine:3.19

ARG POCKETBASE_VERSION=0.22.22

# Optionally inject corporate CA cert for builds behind TLS-intercepting proxies
RUN --mount=type=secret,id=ca_cert,target=/tmp/ca-cert.pem \
    if [ -f /tmp/ca-cert.pem ]; then \
      cat /tmp/ca-cert.pem >> /etc/ssl/certs/ca-certificates.crt; \
    fi

RUN apk add --no-cache \
    ca-certificates \
    wget \
    unzip

# Persist cert after ca-certificates package regenerates the bundle
RUN --mount=type=secret,id=ca_cert,target=/tmp/ca-cert.pem \
    if [ -f /tmp/ca-cert.pem ]; then \
      cp /tmp/ca-cert.pem /usr/local/share/ca-certificates/custom-ca.crt && \
      update-ca-certificates; \
    fi

# Download and install PocketBase
RUN wget https://github.com/pocketbase/pocketbase/releases/download/v${POCKETBASE_VERSION}/pocketbase_${POCKETBASE_VERSION}_linux_amd64.zip \
    && unzip pocketbase_${POCKETBASE_VERSION}_linux_amd64.zip -d /pb/ \
    && rm pocketbase_${POCKETBASE_VERSION}_linux_amd64.zip

# Create data directory
RUN mkdir -p /pb_data /pb_migrations

WORKDIR /pb

EXPOSE 8090

CMD ["/pb/pocketbase", "serve", "--http=0.0.0.0:8090", "--dir=/pb_data", "--migrationsDir=/pb_migrations"]
