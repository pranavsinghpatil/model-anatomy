import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { ArrowUpRight, MessageSquare, Send } from "lucide-react";
import { api } from "../api/client";
import type { Citation, QuestionAnswer } from "../api/contract";
import { ErrorNotice, Loading, Warnings } from "./Common";

type Exchange = { question: string; result: QuestionAnswer; scope: string };
export function Questions({
  analysisId,
  graphId,
  nodeId,
  nodeLabel,
  available,
  ready,
  disabledReason,
  jump,
}: {
  analysisId: string;
  graphId?: string;
  nodeId?: string;
  nodeLabel?: string;
  available?: boolean;
  ready: boolean;
  disabledReason?: string;
  jump: (citation: Citation) => Promise<void>;
}) {
  const [question, setQuestion] = useState("");
  const [exchanges, setExchanges] = useState<Exchange[]>([]);
  const [jumping, setJumping] = useState<string | null>(null);
  const [jumpError, setJumpError] = useState<Error | null>(null);
  const ask = useMutation({
    mutationFn: (input: {
      question: string;
      graphId?: string;
      nodeId?: string;
      scope: string;
    }) => api.question(analysisId, input.question, input.graphId, input.nodeId),
    onSuccess: (result, input) => {
      setExchanges((items) => [
        ...items,
        { question: input.question, result, scope: input.scope },
      ]);
      setQuestion("");
    },
  });
  const disabled =
    Boolean(disabledReason) || available === false || !ready || ask.isPending;
  return (
    <div className="questions" data-testid="questions">
      <div className="question-context">
        <MessageSquare size={15} />
        <span>{nodeLabel ?? (graphId ? "Current graph" : "Model")}</span>
      </div>
      <div className="question-thread">
        {!exchanges.length && (
          <div className="panel-empty">
            <MessageSquare size={26} />
            <strong>Questions about this model</strong>
            <p>
              {disabledReason
                ? disabledReason
                : available === false
                  ? "AI service unavailable"
                  : !ready
                    ? "Waiting for analysis"
                    : "No questions yet"}
            </p>
          </div>
        )}
        {exchanges.map((exchange, index) => (
          <article className="exchange" key={index}>
            <div className="user-question">
              <span className="eyebrow">{exchange.scope}</span>
              <p>{exchange.question}</p>
            </div>
            <div className="answer">
              <div className="answer-label">
                <span className="ai-label">AI generated</span>
                <span>{exchange.result.model}</span>
              </div>
              <p>{exchange.result.answer}</p>
              {exchange.result.citations.length > 0 && (
                <div className="citations">
                  <h3>Evidence</h3>
                  {exchange.result.citations.map((citation) => (
                    <button
                      key={citation.id}
                      className="citation-button"
                      disabled={jumping !== null}
                      onClick={() => {
                        setJumpError(null);
                        setJumping(citation.id);
                        void jump(citation)
                          .catch((error) =>
                            setJumpError(
                              error instanceof Error
                                ? error
                                : new Error("Unable to locate citation."),
                            ),
                          )
                          .finally(() => setJumping(null));
                      }}
                      title={citation.locator}
                    >
                      <span className="citation-id">{citation.id}</span>
                      <span>
                        <strong>{citation.source}</strong>
                        <code>{citation.locator}</code>
                      </span>
                      <ArrowUpRight size={14} />
                    </button>
                  ))}
                </div>
              )}
              <Warnings warnings={exchange.result.warnings} />
            </div>
          </article>
        ))}
        {ask.isPending && <Loading text="Answering from model evidence" />}
        {ask.error && (
          <ErrorNotice
            error={ask.error}
            retry={() => {
              if (ask.variables) ask.mutate(ask.variables);
            }}
          />
        )}
        {jumping && <Loading text="Locating cited evidence" />}
        {jumpError && <ErrorNotice error={jumpError} />}
      </div>
      <form
        className="question-form"
        onSubmit={(event) => {
          event.preventDefault();
          if (question.trim() && !disabled)
            ask.mutate({
              question: question.trim(),
              graphId,
              nodeId,
              scope: nodeLabel ?? "Current graph",
            });
        }}
      >
        <label className="sr-only" htmlFor="question-input">
          Ask about model evidence
        </label>
        <textarea
          id="question-input"
          placeholder={
            disabledReason
              ? disabledReason
              : available === false
                ? "AI service unavailable"
                : "Ask about this model..."
          }
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          disabled={disabled}
          maxLength={2000}
          rows={3}
        />
        <div className="question-form-footer">
          <span>
            {disabledReason
              ? disabledReason
              : available === undefined
                ? "AI availability unknown"
                : available
                  ? "Evidence-bound answers"
                  : "AI unavailable"}
          </span>
          <button
            className="send-button"
            type="submit"
            title="Send question"
            aria-label="Send question"
            disabled={disabled || !question.trim()}
          >
            <Send size={15} />
          </button>
        </div>
      </form>
    </div>
  );
}
