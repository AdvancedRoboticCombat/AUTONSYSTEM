import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "path";

// The backend (main.py) serves the built page (npm run build -> dist/) itself.
// `npm run dev` is only for working on the page: hot reload on
// http://localhost:5173, with the live link passed on to a running backend.
export default defineConfig({
  server: {
    host: "127.0.0.1",
    port: 5173,
    strictPort: true,
    hmr: {
      overlay: false,
    },
    proxy: {
      "/ws": { target: "ws://127.0.0.1:8080", ws: true },
    },
  },
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
});
