import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  return {
    plugins: [react()],
    server: {
      watch: {
        ignored: [
          "**/test-results/**",
          "**/test-results-live/**",
          "**/playwright-report/**",
          "**/.logs/**",
          "**/dist/**",
        ],
      },
      proxy: {
        "/health": {
          target: env.API_PROXY_TARGET || "http://127.0.0.1:8001",
          changeOrigin: true,
        },
        "/api": {
          target: env.API_PROXY_TARGET || "http://127.0.0.1:8001",
          changeOrigin: true,
        },
      },
    },
  };
});
