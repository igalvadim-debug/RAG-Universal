import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
      '@frontend': path.resolve(__dirname, './src/frontend'),
      '@shared': path.resolve(__dirname, './src/shared'),
    },
  },
  server: {
    port: 7960,
    host: true,          // 0.0.0.0 = Zugriff aus dem lokalen Netzwerk
    allowedHosts: true,  // Vite blockt fremde Host-Header sonst im LAN
    proxy: {
      // /api an das Node-Backend (start.bat -> npm run start:backend, Port 7961)
      '/api': 'http://127.0.0.1:7961',
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
    rollupOptions: {
      input: {
        main: path.resolve(__dirname, 'index.html'),
      },
      external: ['fs', 'path', 'child_process', 'chromadb', '@xenova/transformers'],
    },
  },
});