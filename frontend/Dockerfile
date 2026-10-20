# Build the React app, then serve it with nginx, which also proxies /api
# to the backend -- so the browser only ever talks to one origin and the
# refresh-token cookie stays first-party.
FROM node:22-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:1.27-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
