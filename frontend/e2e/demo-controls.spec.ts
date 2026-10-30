import { expect, test, type Page } from "@playwright/test";
import { mockService } from "./service";
import { openAnalysis } from "./helpers";

test.skip(
  process.env.VITE_DEMO_CONTROLS !== "true",
  "Run this suite with VITE_DEMO_CONTROLS=true to enable the deployed demo UI.",
);

const stopped = {
  state: "stopped",
  ready: false,
  queued: 0,
  running: 0,
  start_allowed: true,
  stop_allowed: false,
};
const running = {
  ...stopped,
  state: "running",
  ready: true,
  stop_allowed: true,
};

async function controls(page: Page, initial = stopped) {
  await mockService(page);
  let state = initial;
  const actions: string[] = [];
  let refusal = false;
  await page.route("**/api/demo", async (route) => {
    if (route.request().method() === "POST") {
      const action = route.request().postDataJSON().action as string;
      actions.push(action);
      if (action === "stop" && refusal) {
        state = { ...running, queued: 1, stop_allowed: false };
        return route.fulfill({
          status: 409,
          json: {
            ...state,
            error: {
              message:
                "Stop refused: another visitor has a queued or running analysis.",
            },
          },
        });
      }
      state = action === "start" ? running : stopped;
    }
    await route.fulfill({ json: state });
  });
  return {
    actions,
    refuseStop: () => {
      refusal = true;
    },
  };
}

test("a stopped demo page loads without starting and both buttons work", async ({
  page,
}) => {
  const controller = await controls(page);
  await page.goto("/");
  await expect(
    page.getByText("Demo is stopped.", { exact: false }),
  ).toBeVisible();
  expect(controller.actions).toEqual([]);
  await expect(page.getByRole("button", { name: "Stop Demo" })).toBeDisabled();
  await page.getByRole("button", { name: "Start Demo" }).click();
  await expect(
    page.getByText("Demo is running.", { exact: true }),
  ).toBeVisible();
  await expect(page.getByRole("button", { name: "Start Demo" })).toBeDisabled();
  await page.getByRole("button", { name: "Stop Demo" }).click();
  await expect(
    page.getByText("Demo is stopped.", { exact: false }),
  ).toBeVisible();
  expect(controller.actions).toEqual(["start", "stop"]);
  await page.reload();
  await expect(page.getByRole("button", { name: "Start Demo" })).toBeEnabled();
  expect(controller.actions).toEqual(["start", "stop"]);
});

test("another visitor's active analysis disables stop", async ({ page }) => {
  const controller = await controls(page, {
    ...running,
    running: 1,
    stop_allowed: false,
  });
  await page.goto("/catalog");
  await expect(page.getByRole("button", { name: "Stop Demo" })).toBeDisabled();
  await expect(
    page.getByText("An analysis is queued or running. Stop is unavailable."),
  ).toBeVisible();
  expect(controller.actions).toEqual([]);
});

test("a server refusal after the idle display preserves the running demo", async ({
  page,
}) => {
  const controller = await controls(page, running);
  controller.refuseStop();
  await page.goto("/");
  await page.getByRole("button", { name: "Stop Demo" }).click();
  await expect(page.getByRole("alert")).toContainText("Stop refused");
  await expect(page.getByRole("button", { name: "Stop Demo" })).toBeDisabled();
  await expect(
    page.getByText("Demo is running.", { exact: true }),
  ).toBeVisible();
  expect(controller.actions).toEqual(["stop"]);
});

test("an expired start period keeps the page readable and start disabled", async ({
  page,
}) => {
  await controls(page, { ...stopped, start_allowed: false });
  await page.goto("/");
  await expect(
    page.getByText("The demo start period has ended."),
  ).toBeVisible();
  await expect(page.getByRole("button", { name: "Start Demo" })).toBeDisabled();
  await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
  const dimensions = await page.evaluate(() => ({
    width: window.innerWidth,
    content: document.documentElement.scrollWidth,
  }));
  expect(dimensions.content).toBeLessThanOrEqual(dimensions.width);
});

test("controls leave the inspection workspace within the viewport", async ({
  page,
}) => {
  await controls(page, running);
  await openAnalysis(page);
  await expect(page.getByRole("button", { name: "Stop Demo" })).toBeVisible();
  await expect
    .poll(() =>
      page
        .locator(".workspace-shell")
        .evaluate(
          (workspace) =>
            workspace.getBoundingClientRect().bottom - window.innerHeight,
        ),
    )
    .toBeLessThanOrEqual(1);
});
