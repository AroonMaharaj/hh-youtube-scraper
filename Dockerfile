FROM node:22-slim

WORKDIR /app

COPY package.json ./
RUN npm install --omit=dev

COPY scrape-channel.mjs ./

CMD ["node", "scrape-channel.mjs"]
