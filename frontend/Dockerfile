# Stage 1: Build
FROM node:22-bullseye-slim AS builder
WORKDIR /app

# Cache deps
COPY package.json yarn.lock /app/
ENV PUPPETEER_SKIP_DOWNLOAD=true
RUN yarn install

# Build
COPY ./ /app/
RUN yarn build

# Stage 2: Serve with nginx
FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
