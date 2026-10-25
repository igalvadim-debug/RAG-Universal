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
    port: 7860,
    host: true,
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