# Cloud Run image for the Next.js evidence browser. Build from the repository root.
FROM node:22-bookworm-slim AS dependencies
WORKDIR /workspace/web
COPY web/package.json web/package-lock.json ./
RUN npm ci

FROM node:22-bookworm-slim AS build
WORKDIR /workspace/web
COPY --from=dependencies /workspace/web/node_modules ./node_modules
COPY web ./
RUN npm run build

FROM node:22-bookworm-slim AS runtime
ENV NODE_ENV=production \
    PORT=8080 \
    HOSTNAME=0.0.0.0 \
    TRACE_PROJECT_ROOT=/app
WORKDIR /app/web

# Next's standalone server retains the dynamic API routes required by the app.
COPY --from=build /workspace/web/.next/standalone ./
COPY --from=build /workspace/web/.next/static ./.next/static

# The tracked registry is required when gallery assets are staged separately.
# Runtime assets remain absent by default so clean GitHub checkouts still build.
COPY configs /app/configs
RUN mkdir -p /app/work

EXPOSE 8080
CMD ["node", "server.js"]
