/** Preserve explicit checkpoint selection when a visitor retries a failed job. */
const references = new Map<string, string>();
export function rememberSubmission(id: string, reference: string) {
  references.set(id, reference);
  if (references.size > 100) references.delete(references.keys().next().value!);
  try {
    sessionStorage.setItem(`anatomy-submission:${id}`, reference);
  } catch {
    /* The in-memory reference remains usable without browser storage. */
  }
}
export function retryReference(job: {
  id: string;
  model_id: string;
  revision: string | null;
  model_input?: string | null;
}): string {
  if (job.model_input) return job.model_input;
  try {
    const saved =
      references.get(job.id) ??
      sessionStorage.getItem(`anatomy-submission:${job.id}`);
    if (saved) return saved;
  } catch {
    /* Fall back to the report's pinned repository identity. */
  }
  return job.revision
    ? `https://huggingface.co/${job.model_id.split("/").map(encodeURIComponent).join("/")}/tree/${encodeURIComponent(job.revision)}`
    : job.model_id;
}
