import * as Toast from "@radix-ui/react-toast";
import "./feedback.css";
import { X } from "lucide-react";
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type ReactNode,
  type CSSProperties,
} from "react";
import { Link, useLocation } from "react-router-dom";
import type { AnalysisSummary } from "../api/contract";
import { useRecent } from "./queries";

type FeedbackNotice = {
  id: string;
  title: string;
  description?: string;
  href?: string;
  actionLabel?: string;
};
type VisibleNotice = FeedbackNotice & { instance: number };
type FeedbackApi = {
  notify: (notice: FeedbackNotice) => void;
  dismiss: (id: string) => void;
};
const FeedbackContext = createContext<FeedbackApi | null>(null);

export function useFeedback(): FeedbackApi {
  const context = useContext(FeedbackContext);
  if (!context)
    throw new Error("useFeedback must be used within FeedbackProvider");
  return context;
}

let noticeInstance = 0;

const terminal = (status: string) =>
  !["queued", "running", "cancelling"].includes(status);
const pending = (status: string) => !terminal(status);

function FeedbackViewport({ active }: { active: boolean }) {
  const location = useLocation();
  const [top, setTop] = useState(84);
  useLayoutEffect(() => {
    if (!active) return;
    let frame = 0;
    const measure = () => {
      const zoom =
        Number.parseFloat(getComputedStyle(document.documentElement).zoom) || 1;
      const controls = document.querySelectorAll(
        ".entry-header, .workspace-header, .workspace-viewbar, .inspection-outcome, .graph-header",
      );
      const bottom = Math.max(
        0,
        ...Array.from(controls, (element) => {
          const rect = element.getBoundingClientRect();
          return rect.height > 0 && rect.top < innerHeight ? rect.bottom : 0;
        }),
      );
      setTop((bottom + 12) / zoom);
    };
    const schedule = () => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(measure);
    };
    const resize = new ResizeObserver(schedule);
    resize.observe(document.body);
    const changes = new MutationObserver(schedule);
    changes.observe(document.body, { childList: true, subtree: true });
    window.addEventListener("resize", schedule);
    schedule();
    return () => {
      cancelAnimationFrame(frame);
      resize.disconnect();
      changes.disconnect();
      window.removeEventListener("resize", schedule);
    };
  }, [active, location.key]);
  return (
    <Toast.Viewport
      className="feedback-viewport"
      aria-label="Notifications"
      data-testid="feedback-viewport"
      style={{ "--feedback-top": String(top) + "px" } as CSSProperties}
    />
  );
}

function InspectionNotifications() {
  const { data } = useRecent();
  const { pathname } = useLocation();
  const { notify } = useFeedback();
  const lastStatuses = useRef<Map<string, string> | null>(null);
  const seen = useRef(new Set<string>());

  useEffect(() => {
    const analyses = data?.analyses;
    if (!analyses) return;
    const current = new Map(
      analyses.map((job: AnalysisSummary) => [job.id, job.status]),
    );
    const previous = lastStatuses.current;
    if (previous) {
      for (const job of analyses) {
        const oldStatus = previous.get(job.id);
        if (!oldStatus || !pending(oldStatus) || !terminal(job.status))
          continue;
        const key = `${job.id}:${job.status}`;
        if (seen.current.has(key) || pathname === `/app/analyses/${job.id}`)
          continue;
        seen.current.add(key);
        const outcome =
          job.status === "completed"
            ? "completed"
            : job.status.replaceAll("_", " ");
        notify({
          id: `inspection:${key}`,
          title:
            (
              {
                completed: "Inspection finished",
                partial: "Partial architecture ready",
                inspected: "Files inspected",
                cancelled: "Inspection cancelled",
                failed: "Inspection failed",
                timed_out: "Inspection timed out",
                unsupported: "Inspection unsupported",
              } as Record<string, string>
            )[job.status] ?? `Inspection ${outcome}`,
          description: job.model_id,
          href: `/app/analyses/${encodeURIComponent(job.id)}`,
          actionLabel: "View inspection",
        });
      }
    }
    lastStatuses.current = current;
  }, [data, notify, pathname]);
  return null;
}

export function FeedbackProvider({ children }: { children: ReactNode }) {
  const [notices, setNotices] = useState<VisibleNotice[]>([]);
  const dismiss = useCallback((id: string) => {
    setNotices((items) => items.filter((item) => item.id !== id));
  }, []);
  const notify = useCallback((notice: FeedbackNotice) => {
    noticeInstance += 1;
    const next = { ...notice, instance: noticeInstance };
    setNotices((items) =>
      [...items.filter((item) => item.id !== notice.id), next].slice(-2),
    );
  }, []);
  const api = useRef<FeedbackApi>({ notify, dismiss });
  api.current = { notify, dismiss };

  return (
    <FeedbackContext.Provider value={api.current}>
      <Toast.Provider duration={6000} swipeDirection="right">
        {children}
        <InspectionNotifications />
        <FeedbackViewport active={notices.length > 0} />
        {notices.map((notice) => (
          <Toast.Root
            key={`${notice.id}:${notice.instance}`}
            className="feedback-toast"
            data-testid="feedback-toast"
            open
            onOpenChange={(open) => !open && dismiss(notice.id)}
          >
            <div className="feedback-toast-copy">
              <Toast.Title className="feedback-toast-title">
                {notice.title}
              </Toast.Title>
              {notice.description && (
                <Toast.Description className="feedback-toast-description">
                  {notice.description}
                </Toast.Description>
              )}
            </div>
            {notice.href && notice.actionLabel && (
              <Toast.Action altText={notice.actionLabel} asChild>
                <Link className="feedback-toast-action" to={notice.href}>
                  {notice.actionLabel}
                </Link>
              </Toast.Action>
            )}
            <Toast.Close
              className="feedback-toast-close"
              aria-label="Dismiss notification"
            >
              <X size={16} />
            </Toast.Close>
          </Toast.Root>
        ))}
      </Toast.Provider>
    </FeedbackContext.Provider>
  );
}
