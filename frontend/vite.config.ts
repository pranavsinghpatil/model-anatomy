import { realpathSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { defineConfig, loadEnv, searchForWorkspaceRoot } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  return {
    plugins: [react(), tailwindcss()],
    cacheDir: ".local/vite-cache",
    server: {
      fs: {
        allow: [
          searchForWorkspaceRoot(fileURLToPath(new URL(".", import.meta.url))),
          ...["inter", "jetbrains-mono"].map((font) =>
            realpathSync(
              new URL(
                `./node_modules/@fontsource-variable/${font}`,
                import.meta.url,
              ),
            ),
          ),
        ],
      },
      watch: {
        ignored: [
          "**/test-results*/**",
          "**/public/licenses/**",
          "**/playwright-report/**",
          "**/.logs/**",
          "**/dist/**",
        ],
      },
      proxy: {
        "/health": {
          target: env.API_PROXY_TARGET || "http://127.0.0.1:8000",
          changeOrigin: true,
        },
        "/api": {
          target: env.API_PROXY_TARGET || "http://127.0.0.1:8000",
          changeOrigin: true,
        },
      },
    },
  };
});
