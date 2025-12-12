"use client";

import { useState, useEffect, useRef } from "react";
import ReactMarkdown from "react-markdown";

interface ArticleData {
  title: string;
  content: string;
  sources: string[];
  articleId: string;
  appliedPreferences?: { type: string; text: string; rating: number }[];
  cached?: boolean;
}

interface ImageData {
  bannerUrl?: string;
}

interface AgentStep {
  id: string;
  step: string;
  status: "running" | "done" | "error";
  message: string;
  detail?: string;
  results?: string[];
  expanded?: boolean;
}

interface AgentDecision {
  id: string;
  decision: string;
  reasoning: string;
  action: string;
}

// Step labels for the agent process
const STEP_LABELS: Record<string, { label: string; icon: string }> = {
  cache: { label: "Cache Lookup", icon: "📦" },
  search_a: { label: "Search Topic A", icon: "🔍" },
  search_b: { label: "Search Topic B", icon: "🔍" },
  connections: { label: "Find Connections", icon: "🔗" },
  linkup: { label: "Web Search", icon: "🌐" },
  preferences: { label: "User Preferences", icon: "⭐" },
  context: { label: "Build Context", icon: "📝" },
  generate: { label: "Generate Article", icon: "✨" },
  store: { label: "Save Article", icon: "💾" },
  image: { label: "Generate Image", icon: "🖼️" },
};

export default function Home() {
  const [topicA, setTopicA] = useState("");
  const [topicB, setTopicB] = useState("");
  const [article, setArticle] = useState<ArticleData | null>(null);
  const [images, setImages] = useState<ImageData>({});
  const [isGenerating, setIsGenerating] = useState(false);
  const [agentSteps, setAgentSteps] = useState<AgentStep[]>([]);
  const [agentDecisions, setAgentDecisions] = useState<AgentDecision[]>([]);
  const [showFeedback, setShowFeedback] = useState(false);
  const [streamedContent, setStreamedContent] = useState("");
  const [isCached, setIsCached] = useState(false);
  const [showDecisions, setShowDecisions] = useState(true);
  const stepsRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (stepsRef.current) {
      stepsRef.current.scrollTop = stepsRef.current.scrollHeight;
    }
  }, [agentSteps, streamedContent]);

  const updateStep = (step: string, status: AgentStep["status"], message: string, detail?: string, results?: string[]) => {
    setAgentSteps((prev) => {
      const existing = prev.find((s) => s.step === step);
      if (existing) {
        return prev.map((s) =>
          s.step === step ? { ...s, status, message, detail: detail || s.detail, results: results || s.results } : s
        );
      }
      return [...prev, { id: Date.now().toString(), step, status, message, detail, results, expanded: false }];
    });
  };

  const toggleStepExpanded = (stepId: string) => {
    setAgentSteps((prev) =>
      prev.map((s) => (s.id === stepId ? { ...s, expanded: !s.expanded } : s))
    );
  };

  const handleConnect = async () => {
    if (!topicA || !topicB) return;

    setIsGenerating(true);
    setArticle(null);
    setImages({});
    setAgentSteps([]);
    setAgentDecisions([]);
    setStreamedContent("");
    setIsCached(false);

    try {
      const response = await fetch("/api/connect/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ topicA, topicB }),
      });

      if (!response.ok) throw new Error("Failed to connect");

      const reader = response.body?.getReader();
      if (!reader) throw new Error("No reader");

      const decoder = new TextDecoder();
      let buffer = "";
      let fullContent = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            try {
              const data = JSON.parse(line.slice(6));

              switch (data.type) {
                case "decision":
                  setAgentDecisions((prev) => [
                    ...prev,
                    {
                      id: Date.now().toString(),
                      decision: data.decision,
                      reasoning: data.reasoning,
                      action: data.action,
                    },
                  ]);
                  break;
                case "step":
                  updateStep(data.step, data.status, data.message, data.detail, data.results);
                  break;
                case "content":
                  fullContent += data.chunk;
                  setStreamedContent(fullContent);
                  break;
                case "article":
                  const articleData = {
                    articleId: data.articleId,
                    title: data.title,
                    content: data.content,
                    sources: data.sources || [],
                    cached: data.cached,
                  };
                  setArticle(articleData);
                  setIsCached(data.cached || false);
                  if (data.cached) setStreamedContent(data.content);
                  break;
                case "complete":
                  if (data.appliedPreferences) {
                    setArticle((prev) =>
                      prev ? { ...prev, appliedPreferences: data.appliedPreferences } : prev
                    );
                  }
                  generateBannerImage();
                  break;
                case "error":
                  updateStep("error", "error", data.message, data.detail);
                  break;
              }
            } catch (e) {
              console.error("Parse error:", e);
            }
          }
        }
      }
    } catch (error) {
      console.error("Streaming error:", error);
      updateStep("error", "error", "Failed to generate article", "Check console for details");
    } finally {
      setIsGenerating(false);
    }
  };

  const generateBannerImage = async () => {
    updateStep("image", "running", "Generating illustration", "Using Google Imagen API");
    try {
      const res = await fetch("/api/generate-image", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          topicA,
          topicB,
          connectionSummary: `Illustration for ${topicA} and ${topicB}`,
        }),
      });
      const data = await res.json();
      if (data.imageUrl) {
        setImages({ bannerUrl: data.imageUrl });
        updateStep("image", "done", "Image generated");
      } else {
        updateStep("image", "done", "Image skipped");
      }
    } catch {
      updateStep("image", "done", "Image skipped");
    }
  };

  const reset = () => {
    setArticle(null);
    setImages({});
    setAgentSteps([]);
    setAgentDecisions([]);
    setStreamedContent("");
    setIsCached(false);
  };

  return (
    <main className="min-h-screen bg-white text-[#090E1A]">
      {/* Header */}
      <header className="border-b border-[#090E1A]/10">
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
          <button
            onClick={reset}
            className="text-lg font-semibold tracking-tight hover:text-[#DC244C] transition"
          >
            Wiki Connect
          </button>
          <div className="flex items-center gap-6 text-sm font-semibold">
            {/* Qdrant Logo */}
            <div className="flex items-center gap-2">
              <svg viewBox="0 0 173 200" className="w-5 h-6">
                <polygon fill="#DC244C" points="86.6 0 0 50 0 150 86.6 200 119.08 181.25 119.08 143.75 86.6 162.5 32.48 131.25 32.48 68.75 86.6 37.5 140.73 68.75 140.73 193.75 173.21 175 173.21 50 86.6 0"/>
                <polygon fill="#DC244C" points="54.13 81.25 54.13 118.75 86.6 137.5 119.08 118.75 119.08 81.25 86.6 62.5 54.13 81.25"/>
                <polygon fill="#9e0d38" points="119.08 143.75 119.08 181.25 86.6 200 86.6 162.5 119.08 143.75"/>
                <polygon fill="#9e0d38" points="173.21 50 173.21 175 140.73 193.75 140.73 68.75 173.21 50"/>
                <polygon fill="#ff516b" points="173.21 50 140.73 68.75 86.6 37.5 32.48 68.75 0 50 86.6 0 173.21 50"/>
                <polygon fill="#ff516b" points="119.08 81.25 86.6 100 54.13 81.25 86.6 62.5 119.08 81.25"/>
                <polygon fill="#9e0d38" points="119.08 81.25 119.08 118.75 86.6 137.5 86.6 100 119.08 81.25"/>
              </svg>
              <span className="text-[#DC244C] text-base">Qdrant</span>
            </div>
            <span className="text-[#090E1A]/20">×</span>
            {/* Linkup Logo */}
            <div className="flex items-center gap-2">
              <svg viewBox="0 0 24 24" className="w-5 h-5" fill="none" stroke="#038585" strokeWidth="2.5">
                <circle cx="12" cy="12" r="10" />
                <path d="M12 2a10 10 0 0 1 0 20" strokeDasharray="4 2" />
                <path d="M2 12h20" />
                <path d="M12 2c2.5 2.5 4 6 4 10s-1.5 7.5-4 10" />
                <path d="M12 2c-2.5 2.5-4 6-4 10s1.5 7.5 4 10" />
              </svg>
              <span className="text-[#038585] text-base">Linkup</span>
            </div>
            <span className="text-[#090E1A]/20">×</span>
            {/* Google ADK */}
            <div className="flex items-center gap-2">
              <svg viewBox="0 0 24 24" className="w-5 h-5">
                <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
                <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
              </svg>
              <span className="text-[#090E1A]/70 text-base">Google ADK</span>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-6xl mx-auto px-6">
        {/* Hero */}
        {!article && !isGenerating && (
          <div className="py-20 max-w-2xl">
            <h1 className="text-4xl font-semibold tracking-tight mb-4">
              Connect any two topics
            </h1>
            <p className="text-[#090E1A]/50 text-lg mb-8 leading-relaxed">
              Semantic search across <span className="text-[#090E1A] font-medium">35 million Wikipedia articles</span> powered by{" "}
              <span className="inline-flex items-center gap-1">
                <svg viewBox="0 0 173 200" className="w-4 h-5 inline">
                  <polygon fill="#DC244C" points="86.6 0 0 50 0 150 86.6 200 119.08 181.25 119.08 143.75 86.6 162.5 32.48 131.25 32.48 68.75 86.6 37.5 140.73 68.75 140.73 193.75 173.21 175 173.21 50 86.6 0"/>
                  <polygon fill="#DC244C" points="54.13 81.25 54.13 118.75 86.6 137.5 119.08 118.75 119.08 81.25 86.6 62.5 54.13 81.25"/>
                </svg>
                <span className="text-[#DC244C] font-semibold">Qdrant</span>
              </span>{" "}
              with real-time web grounding from{" "}
              <span className="inline-flex items-center gap-1">
                <svg viewBox="0 0 24 24" className="w-4 h-4 inline" fill="none" stroke="#038585" strokeWidth="2.5">
                  <circle cx="12" cy="12" r="9" />
                  <path d="M2 12h20" />
                  <path d="M12 2c2.5 2.5 4 6 4 10s-1.5 7.5-4 10" />
                </svg>
                <span className="text-[#038585] font-semibold">Linkup</span>
              </span>.
              Built with{" "}
              <span className="inline-flex items-center gap-1">
                <svg viewBox="0 0 24 24" className="w-4 h-4 inline">
                  <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                  <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                  <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
                  <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
                </svg>
                <span className="text-[#4285F4] font-semibold">Google ADK</span>
              </span>.
            </p>
          </div>
        )}

        {/* Input */}
        <div className={`${article || isGenerating ? "py-6 border-b border-[#090E1A]/10" : ""}`}>
          <div className="flex flex-col sm:flex-row gap-3 items-end">
            <div className="flex-1 w-full">
              {!article && !isGenerating && (
                <label className="block text-xs font-medium text-[#090E1A]/40 uppercase tracking-wide mb-2">
                  First topic
                </label>
              )}
              <input
                type="text"
                value={topicA}
                onChange={(e) => setTopicA(e.target.value)}
                placeholder="Black Holes"
                disabled={isGenerating}
                onKeyDown={(e) => e.key === "Enter" && handleConnect()}
                className="w-full px-4 py-3 bg-[#090E1A]/[0.03] border border-[#090E1A]/10 rounded-lg text-[#090E1A] placeholder-[#090E1A]/30 focus:outline-none focus:border-[#DC244C] focus:ring-1 focus:ring-[#DC244C] transition disabled:opacity-50"
              />
            </div>

            <div className="hidden sm:flex items-center justify-center w-10 h-10 text-[#090E1A]/20 text-xl">
              +
            </div>

            <div className="flex-1 w-full">
              {!article && !isGenerating && (
                <label className="block text-xs font-medium text-[#090E1A]/40 uppercase tracking-wide mb-2">
                  Second topic
                </label>
              )}
              <input
                type="text"
                value={topicB}
                onChange={(e) => setTopicB(e.target.value)}
                placeholder="Coffee"
                disabled={isGenerating}
                onKeyDown={(e) => e.key === "Enter" && handleConnect()}
                className="w-full px-4 py-3 bg-[#090E1A]/[0.03] border border-[#090E1A]/10 rounded-lg text-[#090E1A] placeholder-[#090E1A]/30 focus:outline-none focus:border-[#DC244C] focus:ring-1 focus:ring-[#DC244C] transition disabled:opacity-50"
              />
            </div>

            <button
              onClick={handleConnect}
              disabled={!topicA || !topicB || isGenerating}
              className="px-6 py-3 bg-[#DC244C] text-white font-medium rounded-lg hover:bg-[#c41f43] disabled:opacity-40 disabled:cursor-not-allowed transition whitespace-nowrap"
            >
              {isGenerating ? "Generating..." : "Connect"}
            </button>
          </div>

          {/* Quick picks */}
          {!article && !isGenerating && (
            <div className="mt-4 flex flex-wrap gap-2">
              {[
                ["Black Holes", "Coffee"],
                ["Ancient Rome", "Bitcoin"],
                ["DNA", "Architecture"],
                ["Jazz", "Mathematics"],
              ].map(([a, b], i) => (
                <button
                  key={i}
                  onClick={() => { setTopicA(a); setTopicB(b); }}
                  className="px-3 py-1.5 text-sm text-[#090E1A]/60 border border-[#090E1A]/10 rounded-full hover:border-[#DC244C]/30 hover:text-[#DC244C] transition"
                >
                  {a} + {b}
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Main content */}
        {(isGenerating || article) && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 py-8">
            {/* Agent Process */}
            <div className="lg:col-span-4 lg:order-2">
              <div className="sticky top-6">
                {/* Toggle for Decisions vs Steps view */}
                <div className="flex items-center gap-2 mb-4">
                  {isGenerating && (
                    <span className="w-2 h-2 rounded-full bg-[#DC244C] animate-pulse" />
                  )}
                  <h3 className="text-xs font-medium text-[#090E1A]/40 uppercase tracking-wide">
                    Agent Process
                  </h3>
                  <div className="ml-auto flex items-center gap-1">
                    {isCached && (
                      <span className="text-xs text-[#038585] font-medium mr-2">Cached</span>
                    )}
                    <button
                      onClick={() => setShowDecisions(true)}
                      className={`px-2 py-1 text-xs rounded ${showDecisions ? 'bg-[#DC244C] text-white' : 'text-[#090E1A]/40 hover:text-[#090E1A]/60'}`}
                    >
                      Decisions
                    </button>
                    <button
                      onClick={() => setShowDecisions(false)}
                      className={`px-2 py-1 text-xs rounded ${!showDecisions ? 'bg-[#DC244C] text-white' : 'text-[#090E1A]/40 hover:text-[#090E1A]/60'}`}
                    >
                      Steps
                    </button>
                  </div>
                </div>

                {/* Agent Decisions Panel */}
                {showDecisions && agentDecisions.length > 0 && (
                  <div className="mb-4 space-y-2">
                    {agentDecisions.map((d) => (
                      <div
                        key={d.id}
                        className="p-3 rounded-lg border border-[#DC244C]/20 bg-[#DC244C]/[0.02]"
                      >
                        <div className="flex items-start gap-2">
                          <span className="text-sm">🧠</span>
                          <div className="flex-1 min-w-0">
                            <p className="text-sm font-semibold text-[#DC244C]">{d.decision}</p>
                            <p className="text-xs text-[#090E1A]/70 mt-1">{d.reasoning}</p>
                            <div className="mt-2 p-2 bg-[#090E1A]/[0.03] rounded text-xs">
                              <span className="text-[#090E1A]/50">Action: </span>
                              <span className="text-[#090E1A]/80">{d.action}</span>
                            </div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}

                {/* Agent Steps Panel */}
                <div ref={stepsRef} className={`space-y-2 max-h-[70vh] overflow-y-auto pr-2 ${showDecisions && agentDecisions.length > 0 ? 'max-h-[40vh]' : ''}`}>
                  {agentSteps.map((step) => {
                    const stepInfo = STEP_LABELS[step.step] || { label: step.step, icon: "•" };
                    const hasDetails = step.detail || (step.results && step.results.length > 0);
                    const isSearchStep = step.step.startsWith("search_") || step.step === "connections";

                    return (
                      <div
                        key={step.id}
                        className={`rounded-lg border transition-all overflow-hidden ${
                          step.status === "running"
                            ? "border-[#DC244C]/30 bg-[#DC244C]/[0.02]"
                            : step.status === "error"
                            ? "border-red-200 bg-red-50"
                            : "border-[#090E1A]/5 bg-[#090E1A]/[0.01]"
                        }`}
                      >
                        {/* Header - Always visible */}
                        <button
                          onClick={() => hasDetails && toggleStepExpanded(step.id)}
                          className={`w-full p-3 flex items-center gap-2 text-left ${hasDetails ? "cursor-pointer hover:bg-[#090E1A]/[0.02]" : ""}`}
                          disabled={!hasDetails}
                        >
                          {/* Status indicator */}
                          {step.status === "running" ? (
                            <div className="w-4 h-4 flex-shrink-0 rounded-full border-2 border-[#DC244C] border-t-transparent animate-spin" />
                          ) : step.status === "error" ? (
                            <span className="w-4 h-4 flex-shrink-0 flex items-center justify-center text-red-500 text-xs font-bold">✕</span>
                          ) : (
                            <span className="w-4 h-4 flex-shrink-0 flex items-center justify-center text-[#038585] text-xs">✓</span>
                          )}

                          {/* Step icon */}
                          <span className="text-sm flex-shrink-0">{stepInfo.icon}</span>

                          {/* Message */}
                          <div className="flex-1 min-w-0">
                            <p className="text-sm font-medium text-[#090E1A] truncate">{step.message}</p>
                          </div>

                          {/* Expand indicator */}
                          {hasDetails && (
                            <svg
                              className={`w-4 h-4 text-[#090E1A]/30 transition-transform flex-shrink-0 ${step.expanded ? "rotate-180" : ""}`}
                              fill="none"
                              viewBox="0 0 24 24"
                              stroke="currentColor"
                            >
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                            </svg>
                          )}
                        </button>

                        {/* Expanded content */}
                        {step.expanded && hasDetails && (
                          <div className="px-3 pb-3 border-t border-[#090E1A]/5">
                            {step.detail && (
                              <p className="text-xs text-[#090E1A]/60 mt-2 font-mono">{step.detail}</p>
                            )}

                            {step.results && step.results.length > 0 && (
                              <div className="mt-3 space-y-2">
                                <p className="text-xs font-medium text-[#090E1A]/40 uppercase tracking-wide">
                                  {isSearchStep ? "Retrieved Documents" : "Results"}
                                </p>
                                {step.results.map((r, i) => (
                                  <div
                                    key={i}
                                    className="p-2 bg-white border border-[#090E1A]/5 rounded text-xs"
                                  >
                                    {r.startsWith("http") ? (
                                      <a
                                        href={r}
                                        target="_blank"
                                        rel="noopener noreferrer"
                                        className="text-[#DC244C] hover:underline break-all"
                                      >
                                        {r}
                                      </a>
                                    ) : r.includes(":") && isSearchStep ? (
                                      <div>
                                        <span className="font-medium text-[#090E1A]">
                                          {r.split(":")[0]}
                                        </span>
                                        <span className="text-[#090E1A]/60">
                                          :{r.split(":").slice(1).join(":")}
                                        </span>
                                      </div>
                                    ) : (
                                      <span className="text-[#090E1A]/70">{r}</span>
                                    )}
                                  </div>
                                ))}
                              </div>
                            )}
                          </div>
                        )}

                        {/* Preview of results (when collapsed) */}
                        {!step.expanded && step.results && step.results.length > 0 && step.status === "done" && (
                          <div className="px-3 pb-2 -mt-1">
                            <p className="text-xs text-[#090E1A]/40 truncate">
                              {isSearchStep ? `${step.results.length} results` : step.results[0]}
                            </p>
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>

            {/* Article */}
            <div className="lg:col-span-8 lg:order-1">
              {/* Banner */}
              {images.bannerUrl && (
                <div className="mb-6 rounded-lg overflow-hidden">
                  <img
                    src={images.bannerUrl}
                    alt=""
                    className="w-full h-48 object-cover"
                  />
                </div>
              )}

              {/* Title */}
              {article && (
                <h1 className="text-3xl font-semibold tracking-tight text-[#090E1A] mb-6">
                  {article.title}
                </h1>
              )}

              {/* Applied Preferences */}
              {article?.appliedPreferences && article.appliedPreferences.length > 0 && (
                <div className="mb-6 p-3 bg-[#038585]/5 border border-[#038585]/20 rounded-lg">
                  <p className="text-sm text-[#038585]">
                    <span className="font-medium">Personalized:</span>{" "}
                    {article.appliedPreferences.slice(0, 2).map((p) => p.text).join("; ")}
                  </p>
                </div>
              )}

              {/* Content */}
              <article className="prose prose-neutral max-w-none">
                <ReactMarkdown
                  components={{
                    h1: ({ children }) => (
                      <h1 className="text-2xl font-semibold text-[#090E1A] mt-8 mb-4">{children}</h1>
                    ),
                    h2: ({ children }) => (
                      <h2 className="text-xl font-semibold text-[#090E1A] mt-6 mb-3">{children}</h2>
                    ),
                    h3: ({ children }) => (
                      <h3 className="text-lg font-medium text-[#090E1A] mt-5 mb-2">{children}</h3>
                    ),
                    p: ({ children }) => (
                      <p className="text-[#090E1A]/80 leading-relaxed mb-4">{children}</p>
                    ),
                    ul: ({ children }) => (
                      <ul className="list-disc list-outside ml-5 text-[#090E1A]/80 mb-4 space-y-1">{children}</ul>
                    ),
                    ol: ({ children }) => (
                      <ol className="list-decimal list-outside ml-5 text-[#090E1A]/80 mb-4 space-y-1">{children}</ol>
                    ),
                    a: ({ href, children }) => (
                      <a href={href} className="text-[#DC244C] hover:underline" target="_blank" rel="noopener noreferrer">
                        {children}
                      </a>
                    ),
                    blockquote: ({ children }) => (
                      <blockquote className="border-l-2 border-[#DC244C] pl-4 italic text-[#090E1A]/60 my-4">
                        {children}
                      </blockquote>
                    ),
                  }}
                >
                  {streamedContent || article?.content || ""}
                </ReactMarkdown>
                {isGenerating && streamedContent && (
                  <span className="inline-block w-0.5 h-5 bg-[#DC244C] animate-pulse ml-0.5" />
                )}
              </article>

              {/* Sources */}
              {article && article.sources.length > 0 && (
                <div className="mt-10 pt-6 border-t border-[#090E1A]/10">
                  <h3 className="text-xs font-medium text-[#090E1A]/40 uppercase tracking-wide mb-3">
                    Sources
                  </h3>
                  <div className="space-y-2">
                    {article.sources.map((source, i) => (
                      <a
                        key={i}
                        href={source}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="block text-sm text-[#090E1A]/60 hover:text-[#DC244C] truncate transition"
                      >
                        [{i + 1}] {source}
                      </a>
                    ))}
                  </div>
                </div>
              )}

              {/* Actions */}
              {article && !isGenerating && (
                <div className="mt-8 flex gap-3">
                  <button
                    onClick={reset}
                    className="px-4 py-2 text-sm text-[#090E1A]/60 border border-[#090E1A]/10 rounded-lg hover:border-[#090E1A]/30 transition"
                  >
                    New connection
                  </button>
                  <button
                    onClick={() => setShowFeedback(true)}
                    className="px-4 py-2 text-sm text-white bg-[#DC244C] rounded-lg hover:bg-[#c41f43] transition"
                  >
                    Give feedback
                  </button>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Features */}
        {!article && !isGenerating && (
          <div className="py-16 border-t border-[#090E1A]/5 mt-8">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
              {/* Qdrant Card */}
              <div className="group p-6 rounded-2xl border border-[#DC244C]/10 bg-gradient-to-br from-[#DC244C]/[0.02] to-transparent hover:border-[#DC244C]/30 hover:shadow-lg hover:shadow-[#DC244C]/5 transition-all duration-300">
                <div className="flex items-center gap-3 mb-4">
                  <svg viewBox="0 0 173 200" className="w-8 h-9">
                    <polygon fill="#DC244C" points="86.6 0 0 50 0 150 86.6 200 119.08 181.25 119.08 143.75 86.6 162.5 32.48 131.25 32.48 68.75 86.6 37.5 140.73 68.75 140.73 193.75 173.21 175 173.21 50 86.6 0"/>
                    <polygon fill="#DC244C" points="54.13 81.25 54.13 118.75 86.6 137.5 119.08 118.75 119.08 81.25 86.6 62.5 54.13 81.25"/>
                    <polygon fill="#9e0d38" points="119.08 143.75 119.08 181.25 86.6 200 86.6 162.5 119.08 143.75"/>
                    <polygon fill="#9e0d38" points="173.21 50 173.21 175 140.73 193.75 140.73 68.75 173.21 50"/>
                    <polygon fill="#ff516b" points="173.21 50 140.73 68.75 86.6 37.5 32.48 68.75 0 50 86.6 0 173.21 50"/>
                  </svg>
                  <h3 className="text-lg font-bold text-[#DC244C]">Qdrant</h3>
                </div>
                <p className="text-[#090E1A]/70 text-sm leading-relaxed mb-4">
                  Vector database powering semantic search across <span className="font-semibold text-[#090E1A]">35M+ Wikipedia articles</span>.
                </p>
                <div className="space-y-2 text-xs">
                  <div className="flex items-center gap-2 text-[#090E1A]/50">
                    <span className="w-1.5 h-1.5 rounded-full bg-[#DC244C]"></span>
                    <span>Cohere multilingual embeddings</span>
                  </div>
                  <div className="flex items-center gap-2 text-[#090E1A]/50">
                    <span className="w-1.5 h-1.5 rounded-full bg-[#DC244C]"></span>
                    <span>6 collections: WORLD, USER_*, LINKUP_CACHE</span>
                  </div>
                  <div className="flex items-center gap-2 text-[#090E1A]/50">
                    <span className="w-1.5 h-1.5 rounded-full bg-[#DC244C]"></span>
                    <span>Sub-second similarity search</span>
                  </div>
                </div>
              </div>

              {/* Linkup Card */}
              <div className="group p-6 rounded-2xl border border-[#038585]/10 bg-gradient-to-br from-[#038585]/[0.02] to-transparent hover:border-[#038585]/30 hover:shadow-lg hover:shadow-[#038585]/5 transition-all duration-300">
                <div className="flex items-center gap-3 mb-4">
                  <div className="w-8 h-8 rounded-lg bg-[#038585] flex items-center justify-center">
                    <svg viewBox="0 0 24 24" className="w-5 h-5" fill="none" stroke="white" strokeWidth="2">
                      <circle cx="12" cy="12" r="10" />
                      <path d="M2 12h20" />
                      <path d="M12 2c2.5 2.5 4 6 4 10s-1.5 7.5-4 10" />
                      <path d="M12 2c-2.5 2.5-4 6-4 10s1.5 7.5 4 10" />
                    </svg>
                  </div>
                  <h3 className="text-lg font-bold text-[#038585]">Linkup</h3>
                </div>
                <p className="text-[#090E1A]/70 text-sm leading-relaxed mb-4">
                  Real-time web search API for <span className="font-semibold text-[#090E1A]">dynamic grounding</span> when Wikipedia falls short.
                </p>
                <div className="space-y-2 text-xs">
                  <div className="flex items-center gap-2 text-[#090E1A]/50">
                    <span className="w-1.5 h-1.5 rounded-full bg-[#038585]"></span>
                    <span>Fresh content for recent topics</span>
                  </div>
                  <div className="flex items-center gap-2 text-[#090E1A]/50">
                    <span className="w-1.5 h-1.5 rounded-full bg-[#038585]"></span>
                    <span>Results cached in Qdrant (24h TTL)</span>
                  </div>
                  <div className="flex items-center gap-2 text-[#090E1A]/50">
                    <span className="w-1.5 h-1.5 rounded-full bg-[#038585]"></span>
                    <span>Automatic fallback routing</span>
                  </div>
                </div>
              </div>

              {/* Google ADK Card */}
              <div className="group p-6 rounded-2xl border border-[#4285F4]/10 bg-gradient-to-br from-[#4285F4]/[0.02] to-transparent hover:border-[#4285F4]/30 hover:shadow-lg hover:shadow-[#4285F4]/5 transition-all duration-300">
                <div className="flex items-center gap-3 mb-4">
                  <svg viewBox="0 0 24 24" className="w-8 h-8">
                    <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                    <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                    <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
                    <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
                  </svg>
                  <h3 className="text-lg font-bold text-[#4285F4]">Google ADK</h3>
                </div>
                <p className="text-[#090E1A]/70 text-sm leading-relaxed mb-4">
                  Agent Development Kit with <span className="font-semibold text-[#090E1A]">Gemini 2.0 Flash</span> for grounded generation.
                </p>
                <div className="space-y-2 text-xs">
                  <div className="flex items-center gap-2 text-[#090E1A]/50">
                    <span className="w-1.5 h-1.5 rounded-full bg-[#4285F4]"></span>
                    <span>Streaming article generation</span>
                  </div>
                  <div className="flex items-center gap-2 text-[#090E1A]/50">
                    <span className="w-1.5 h-1.5 rounded-full bg-[#34A853]"></span>
                    <span>Nano Banana Pro image generation</span>
                  </div>
                  <div className="flex items-center gap-2 text-[#090E1A]/50">
                    <span className="w-1.5 h-1.5 rounded-full bg-[#EA4335]"></span>
                    <span>Transparent agent decisions</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Feedback Modal */}
      {showFeedback && article && (
        <FeedbackModal articleId={article.articleId} onClose={() => setShowFeedback(false)} />
      )}

      {/* Footer */}
      <footer className="border-t border-[#090E1A]/10 mt-20">
        <div className="max-w-6xl mx-auto px-6 py-8">
          <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
            <div className="flex items-center gap-2 text-sm text-[#090E1A]/50">
              <span>Built with</span>
              <a href="https://qdrant.tech" target="_blank" rel="noopener noreferrer" className="text-[#DC244C] hover:underline font-medium">Qdrant</a>
              <span>+</span>
              <a href="https://linkup.so" target="_blank" rel="noopener noreferrer" className="text-[#038585] hover:underline font-medium">Linkup</a>
              <span>+</span>
              <a href="https://google.github.io/adk-docs" target="_blank" rel="noopener noreferrer" className="text-[#4285F4] hover:underline font-medium">Google ADK</a>
            </div>
            <div className="flex items-center gap-5">
              {/* GitHub */}
              <a
                href="https://github.com/thierrypdamiba/wiki-connect"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-1.5 text-[#090E1A]/40 hover:text-[#090E1A] transition"
              >
                <svg viewBox="0 0 24 24" className="w-4 h-4" fill="currentColor">
                  <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
                </svg>
                <span className="text-xs">GitHub</span>
              </a>
              {/* Qdrant */}
              <a
                href="https://qdrant.tech"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-1.5 text-[#090E1A]/40 hover:text-[#DC244C] transition"
              >
                <svg viewBox="0 0 173 200" className="w-4 h-5" fill="currentColor">
                  <polygon points="86.6 0 0 50 0 150 86.6 200 119.08 181.25 119.08 143.75 86.6 162.5 32.48 131.25 32.48 68.75 86.6 37.5 140.73 68.75 140.73 193.75 173.21 175 173.21 50 86.6 0"/>
                  <polygon points="54.13 81.25 54.13 118.75 86.6 137.5 119.08 118.75 119.08 81.25 86.6 62.5 54.13 81.25"/>
                </svg>
                <span className="text-xs">Qdrant</span>
              </a>
              {/* Linkup */}
              <a
                href="https://linkup.so"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-1.5 text-[#090E1A]/40 hover:text-[#038585] transition"
              >
                <svg viewBox="0 0 24 24" className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="2">
                  <circle cx="12" cy="12" r="10" />
                  <path d="M2 12h20" />
                  <path d="M12 2c2.5 2.5 4 6 4 10s-1.5 7.5-4 10" />
                  <path d="M12 2c-2.5 2.5-4 6-4 10s1.5 7.5 4 10" />
                </svg>
                <span className="text-xs">Linkup</span>
              </a>
              {/* Google ADK */}
              <a
                href="https://google.github.io/adk-docs"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-1.5 text-[#090E1A]/40 hover:text-[#4285F4] transition"
              >
                <svg viewBox="0 0 24 24" className="w-4 h-4">
                  <path fill="currentColor" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                  <path fill="currentColor" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                  <path fill="currentColor" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
                  <path fill="currentColor" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
                </svg>
                <span className="text-xs">Google ADK</span>
              </a>
            </div>
          </div>
        </div>
      </footer>
    </main>
  );
}

function FeedbackModal({ articleId, onClose }: { articleId: string; onClose: () => void }) {
  const [feedbackType, setFeedbackType] = useState("text_style");
  const [feedbackText, setFeedbackText] = useState("");
  const [rating, setRating] = useState(4);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitted, setSubmitted] = useState(false);

  const handleSubmit = async () => {
    if (!feedbackText.trim()) return;
    setIsSubmitting(true);

    try {
      await fetch("/api/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ articleId, feedbackType, feedbackText, rating }),
      });
      setSubmitted(true);
      setTimeout(onClose, 1500);
    } catch (error) {
      console.error("Error:", error);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-[#090E1A]/50">
      <div className="bg-white rounded-xl w-full max-w-md overflow-hidden shadow-2xl">
        <div className="px-6 py-4 border-b border-[#090E1A]/10 flex items-center justify-between">
          <h3 className="font-semibold text-[#090E1A]">Improve future articles</h3>
          <button onClick={onClose} className="text-[#090E1A]/40 hover:text-[#090E1A] transition">
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {submitted ? (
          <div className="p-8 text-center">
            <div className="w-12 h-12 rounded-full bg-[#038585]/10 flex items-center justify-center mx-auto mb-4">
              <span className="text-[#038585] text-xl">✓</span>
            </div>
            <p className="font-medium text-[#090E1A]">Thanks for your feedback</p>
            <p className="text-sm text-[#090E1A]/60 mt-1">It will be applied to future articles.</p>
          </div>
        ) : (
          <div className="p-6 space-y-4">
            <div>
              <label className="block text-xs font-medium text-[#090E1A]/40 uppercase tracking-wide mb-2">
                Type
              </label>
              <select
                value={feedbackType}
                onChange={(e) => setFeedbackType(e.target.value)}
                className="w-full px-3 py-2 bg-[#090E1A]/[0.03] border border-[#090E1A]/10 rounded-lg text-[#090E1A] focus:outline-none focus:border-[#DC244C]"
              >
                <option value="text_style">Writing Style</option>
                <option value="format">Format</option>
                <option value="content">Content Depth</option>
              </select>
            </div>

            <div>
              <label className="block text-xs font-medium text-[#090E1A]/40 uppercase tracking-wide mb-2">
                Rating
              </label>
              <div className="flex gap-1">
                {[1, 2, 3, 4, 5].map((star) => (
                  <button
                    key={star}
                    onClick={() => setRating(star)}
                    className={`text-xl transition ${star <= rating ? "text-[#DC244C]" : "text-[#090E1A]/20"}`}
                  >
                    ★
                  </button>
                ))}
              </div>
            </div>

            <div>
              <label className="block text-xs font-medium text-[#090E1A]/40 uppercase tracking-wide mb-2">
                Feedback
              </label>
              <textarea
                value={feedbackText}
                onChange={(e) => setFeedbackText(e.target.value)}
                placeholder="e.g., Use more analogies, shorter paragraphs..."
                rows={3}
                className="w-full px-3 py-2 bg-[#090E1A]/[0.03] border border-[#090E1A]/10 rounded-lg text-[#090E1A] placeholder-[#090E1A]/30 focus:outline-none focus:border-[#DC244C]"
              />
            </div>

            <button
              onClick={handleSubmit}
              disabled={!feedbackText.trim() || isSubmitting}
              className="w-full py-3 bg-[#DC244C] text-white font-medium rounded-lg hover:bg-[#c41f43] disabled:opacity-40 transition"
            >
              {isSubmitting ? "Submitting..." : "Submit"}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
