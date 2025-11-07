import { defineConfig } from 'vite'
import mkcert from 'vite-plugin-mkcert'

export default defineConfig({
  server: {
    https: true,      // enables HTTPS
    host: '0.0.0.0',  // allows access from other devices (e.g. iPhone)
  },
  plugins: [mkcert()],
})
