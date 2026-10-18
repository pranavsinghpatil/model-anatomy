import { writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import openapiTS, { astToString, COMMENT_HEADER } from "openapi-typescript";
import { format, resolveConfig } from "prettier";

const input = process.argv[2] ?? "http://127.0.0.1:8001/openapi.json";
const parsed = URL.canParse(input) ? new URL(input) : null;
const source =
  parsed && ["http:", "https:", "file:"].includes(parsed.protocol)
    ? parsed
    : pathToFileURL(resolve(input));
const output = fileURLToPath(
  new URL("../src/api/generated.ts", import.meta.url),
);
const schema = await openapiTS(source);
const code = await format(`${COMMENT_HEADER}${astToString(schema)}`, {
  ...(await resolveConfig(output)),
  parser: "typescript",
});
await writeFile(output, code, "utf8");
process.stdout.write("Generated and formatted actual OpenAPI types.\n");
