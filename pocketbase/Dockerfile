FROM alpine:3.19

ARG POCKETBASE_VERSION=0.22.22

RUN apk add --no-cache \
    ca-certificates \
    wget \
    unzip

# Download and install PocketBase
RUN wget https://github.com/pocketbase/pocketbase/releases/download/v${POCKETBASE_VERSION}/pocketbase_${POCKETBASE_VERSION}_linux_amd64.zip \
    && unzip pocketbase_${POCKETBASE_VERSION}_linux_amd64.zip -d /pb/ \
    && rm pocketbase_${POCKETBASE_VERSION}_linux_amd64.zip

# Create data directory
RUN mkdir -p /pb_data /pb_migrations

WORKDIR /pb

EXPOSE 8090

CMD ["/pb/pocketbase", "serve", "--http=0.0.0.0:8090", "--dir=/pb_data", "--migrationsDir=/pb_migrations"]
