"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { ChevronRight, ChevronDown, Home, MessageCircle, Volume2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { updateScreenplayMetadata, generateFirstSceneAudio } from "@/lib/api";
import { Screenplay } from "@/types/screenplay";
import HorizontalNav from "./HorizontalNav";
import MarginAssistant from "./MarginAssistant";
import HighlightAskButton from "./HighlightAskButton";
import VerticalNav from "./VerticalNav";
import ThemeToggle from "./ThemeToggle";
import GhostInput from "./GhostInput";

const SM_BREAKPOINT = 640;

function useIsSm() {
  const [isSm, setIsSm] = useState(true);
  useEffect(() => {
    const mq = window.matchMedia(`(min-width: ${SM_BREAKPOINT}px)`);
    const set = () => setIsSm(mq.matches);
    set();
    mq.addEventListener("change", set);
    return () => mq.removeEventListener("change", set);
  }, []);
  return isSm;
}

type NavMode = "horizontal" | "vertical";

interface Props {
  screenplay: Screenplay;
  screenplayId: string;
  onReset: () => void;
}

export default function ScreenplayViewer({
  screenplay,
  screenplayId,
  onReset,
}: Props) {
  const isSm = useIsSm();
  const [mode, setMode] = useState<NavMode>("horizontal");
  const [currentSceneIndex, setCurrentSceneIndex] = useState(0);
  const [marginOpen, setMarginOpen] = useState(false);
  const [activeReferenceText, setActiveReferenceText] = useState<string | null>(
    null,
  );
  const [initialReference, setInitialReference] = useState<string | null>(null);
  const [pendingNavigation, setPendingNavigation] = useState<{
    sceneIndex: number;
    key: number;
  } | null>(null);
  const [selection, setSelection] = useState<{ text: string; rect: DOMRect } | null>(
    null,
  );
  const [title, setTitle] = useState(screenplay.title);
  const [authors, setAuthors] = useState(screenplay.authors);
  const [generateModalOpen, setGenerateModalOpen] = useState(false);
  const [audioGenerating, setAudioGenerating] = useState(false);
  const [generateError, setGenerateError] = useState<string | null>(null);
  /** Ref to the scrollable scene body only (excludes header + nav footer). Used to constrain selection and button placement. */
  const sceneContentRef = useRef<HTMLDivElement>(null);

  const commitTitle = useCallback(
    (newTitle: string) => {
      const upper = newTitle.trim().toUpperCase();
      setTitle(upper);
      updateScreenplayMetadata(screenplayId, { title: upper }).catch(() => {
        setTitle(screenplay.title);
      });
    },
    [screenplayId, screenplay.title],
  );

  const commitAuthors = useCallback(
    (raw: string) => {
      const parsed = raw
        .split(/,|&/)
        .map((s) => s.trim())
        .filter(Boolean);
      setAuthors(parsed);
      updateScreenplayMetadata(screenplayId, { authors: parsed }).catch(() => {
        setAuthors(screenplay.authors);
      });
    },
    [screenplayId, screenplay.authors],
  );

  const handleSelectionChange = useCallback(() => {
    const sel = window.getSelection();
    if (!sel || sel.isCollapsed) {
      setSelection(null);
      return;
    }
    const text = sel.toString().trim();
    if (!text) {
      setSelection(null);
      return;
    }
    if (sceneContentRef.current && !sceneContentRef.current.contains(sel.anchorNode)) {
      setSelection(null);
      return;
    }
    try {
      const rect = sel.getRangeAt(0).getBoundingClientRect();
      setSelection({ text, rect });
    } catch {
      setSelection(null);
    }
  }, []);

  useEffect(() => {
    document.addEventListener("selectionchange", handleSelectionChange);
    return () => document.removeEventListener("selectionchange", handleSelectionChange);
  }, [handleSelectionChange]);

  const handleAskFromHighlight = useCallback(() => {
    if (!selection) return;
    setInitialReference(selection.text);
    setMarginOpen(true);
    setSelection(null);
    window.getSelection()?.removeAllRanges();
  }, [selection]);

  const clearInitialReference = useCallback(() => setInitialReference(null), []);

  const closeMargin = useCallback(() => {
    setMarginOpen(false);
    setActiveReferenceText(null);
  }, []);

  const clearActiveReference = useCallback(() => setActiveReferenceText(null), []);

  const handleNavigateToScene = useCallback(
    (sceneIndex: number, quote?: string) => {
      if (sceneIndex < 0 || sceneIndex >= screenplay.scenes.length) return;
      setPendingNavigation({ sceneIndex, key: Date.now() });
      if (quote) {
        setActiveReferenceText(quote);
      }
    },
    [screenplay.scenes.length],
  );

  const handleGenerateAudioClick = useCallback(() => {
    setGenerateError(null);
    setGenerateModalOpen(true);
  }, []);

  const closeGenerateModal = useCallback(() => {
    if (!audioGenerating) {
      setGenerateModalOpen(false);
      setGenerateError(null);
    }
  }, [audioGenerating]);

  const handleApproveAudioGeneration = useCallback(async () => {
    setAudioGenerating(true);
    setGenerateError(null);
    try {
      const blob = await generateFirstSceneAudio(screenplayId);
      setGenerateModalOpen(false);
      const url = URL.createObjectURL(blob);
      const audio = new Audio(url);
      audio.addEventListener("ended", () => URL.revokeObjectURL(url));
      await audio.play();
    } catch (err) {
      setGenerateError(err instanceof Error ? err.message : "Failed to generate audio");
    } finally {
      setAudioGenerating(false);
    }
  }, [screenplayId]);

  return (
    <div className="flex h-screen flex-col bg-background">
      <header className="flex flex-col gap-3 border-b border-border bg-background px-4 py-3 sm:flex-row sm:flex-wrap sm:items-center sm:gap-3 sm:px-6 sm:py-3">
        {/* Row 1 on mobile: title + author (full width) — inline editable */}
        <div className="min-w-0 flex-1">
          <div className="font-mono text-base font-bold leading-tight text-foreground sm:text-sm">
            <GhostInput
              value={title}
              placeholder="Untitled Screenplay"
              onCommit={commitTitle}
              showPencilHint
              className="font-mono text-base font-bold leading-tight text-foreground sm:text-sm"
            />
          </div>
          <div className="mt-0.5 font-mono text-xs text-muted-foreground">
            <span className="pointer-events-none select-none">by </span>
            <GhostInput
              value={authors.join(", ")}
              placeholder="Add Author"
              onCommit={commitAuthors}
              className="font-mono text-xs text-muted-foreground"
              hoverClassName="text-foreground"
            />
          </div>
        </div>

        {/* Row 2 on mobile: stats */}
        <div className="flex items-center justify-between gap-4 sm:order-2 sm:w-auto sm:justify-end">
          <span className="font-mono text-xs text-muted-foreground/80">
            {screenplay.scenes.length} scenes
          </span>
          <span className="font-mono text-xs text-muted-foreground/80">
            {screenplay.all_characters.length} characters
          </span>
        </div>

        {/* Row 3 on mobile: toolbar — Ask first for visibility, then view/look/actions */}
        <div className="flex flex-wrap items-center gap-2 sm:order-3 sm:w-auto">
          {/* Margin assistant / Ask — primary action, always labeled */}
          <button
            type="button"
            onClick={() => (marginOpen ? closeMargin() : setMarginOpen(true))}
            className={cn(
              "flex min-h-[44px] items-center gap-2 rounded-lg border px-3 font-mono text-xs font-medium transition-colors sm:min-h-0 sm:py-1.5",
              marginOpen
                ? "border-primary bg-primary text-primary-foreground"
                : "border-primary bg-primary/10 text-primary hover:bg-primary/20",
            )}
            aria-label={marginOpen ? "Close assistant" : "Open assistant — ask questions about the screenplay"}
          >
            <MessageCircle className="h-4 w-4 shrink-0" aria-hidden />
            <span>Ask</span>
          </button>

          {/* Nav mode toggle */}
          <div className="flex rounded-lg border border-border bg-card p-0.5">
            <button
              onClick={() => setMode("horizontal")}
              className={cn(
                "flex min-h-[44px] min-w-[44px] items-center justify-center gap-1 rounded-md font-mono text-xs transition-colors sm:min-h-0 sm:min-w-0 sm:px-3 sm:py-1.5 sm:gap-1.5",
                mode === "horizontal"
                  ? "bg-accent text-accent-foreground"
                  : "text-muted-foreground hover:text-foreground",
              )}
              aria-label="Horizontal view"
            >
              <ChevronRight className="h-3.5 w-3.5" />
              <span className="hidden sm:inline">Horizontal</span>
              <span className="sm:hidden">H</span>
            </button>
            <button
              onClick={() => setMode("vertical")}
              className={cn(
                "flex min-h-[44px] min-w-[44px] items-center justify-center gap-1 rounded-md font-mono text-xs transition-colors sm:min-h-0 sm:min-w-0 sm:px-3 sm:py-1.5 sm:gap-1.5",
                mode === "vertical"
                  ? "bg-accent text-accent-foreground"
                  : "text-muted-foreground hover:text-foreground",
              )}
              aria-label="Vertical view"
            >
              <ChevronDown className="h-3.5 w-3.5" />
              <span className="hidden sm:inline">Vertical</span>
              <span className="sm:hidden">V</span>
            </button>
          </div>

          <button
            type="button"
            onClick={handleGenerateAudioClick}
            className={cn(
              "flex min-h-[44px] items-center gap-2 rounded-lg border border-border px-3 font-mono text-xs",
              "transition-colors hover:bg-accent hover:text-accent-foreground sm:min-h-0 sm:py-1.5",
            )}
            aria-label="Generate dialogue audio — estimate cost and generate"
          >
            <Volume2 className="h-3.5 w-3.5 shrink-0" aria-hidden />
            <span className="hidden sm:inline">Generate audio</span>
            <span className="sm:hidden">Audio</span>
          </button>

          <div className="flex items-center gap-2">
            <ThemeToggle />
          </div>

          <button
            onClick={onReset}
            className={cn(
              "flex min-h-[44px] items-center gap-1.5 rounded-lg border border-border px-3 font-mono text-xs text-muted-foreground",
              "transition-colors hover:bg-accent hover:text-accent-foreground sm:min-h-0 sm:py-1.5",
            )}
            aria-label="Go to main page"
          >
            <Home className="h-3.5 w-3.5 shrink-0" />
            <span className="hidden sm:inline">Main page</span>
            <span className="sm:hidden">Home</span>
          </button>
        </div>
      </header>

      {/* Dual-pane: content (left/center) + context margin (right, desktop only) */}
      <div className="flex flex-1 min-h-0">
        <div
          className={cn(
            "flex min-w-0 flex-1 flex-col overflow-hidden",
            marginOpen && isSm && "border-r border-border",
          )}
        >
          {mode === "horizontal" ? (
            <HorizontalNav
              scenes={screenplay.scenes}
              activeReferenceText={activeReferenceText}
              onClearActiveReference={clearActiveReference}
              sceneContentRef={sceneContentRef}
              navigateToScene={pendingNavigation}
              initialSceneIndex={currentSceneIndex}
              onSceneChange={setCurrentSceneIndex}
              screenplayId={screenplayId}
            />
          ) : (
            <VerticalNav
              scenes={screenplay.scenes}
              activeReferenceText={activeReferenceText}
              onClearActiveReference={clearActiveReference}
              sceneContentRef={sceneContentRef}
              navigateToScene={pendingNavigation}
              initialSceneIndex={currentSceneIndex}
              onSceneChange={setCurrentSceneIndex}
              screenplayId={screenplayId}
            />
          )}
        </div>

        {/* Desktop: side panel */}
        {marginOpen && isSm && (
          <MarginAssistant
            screenplayId={screenplayId}
            screenplayTitle={title}
            scenes={screenplay.scenes}
            open={marginOpen}
            onClose={closeMargin}
            initialReference={initialReference}
            onInitialReferenceConsumed={clearInitialReference}
            onActiveReferenceChange={setActiveReferenceText}
            onNavigateToScene={handleNavigateToScene}
          />
        )}
      </div>

      {/* Small screens: popover overlay (bottom sheet) */}
      {marginOpen && !isSm && (
        <div
          className="fixed inset-0 z-50 flex flex-col sm:hidden"
          role="dialog"
          aria-modal="true"
          aria-label="Ask about this screenplay"
        >
          <button
            type="button"
            className="absolute inset-0 bg-black/50 backdrop-blur-[2px]"
            onClick={closeMargin}
            aria-label="Close"
          />
          <div className="relative mt-auto flex max-h-[90vh] flex-col overflow-hidden rounded-t-xl border-t border-border bg-background shadow-xl">
            <MarginAssistant
              screenplayId={screenplayId}
              screenplayTitle={title}
              scenes={screenplay.scenes}
              open={marginOpen}
              onClose={closeMargin}
              initialReference={initialReference}
              onInitialReferenceConsumed={clearInitialReference}
              onActiveReferenceChange={setActiveReferenceText}
              onNavigateToScene={handleNavigateToScene}
              variant="popover"
            />
          </div>
        </div>
      )}

      {selection && (
        <HighlightAskButton
          rect={selection.rect}
          contentBoundsRef={sceneContentRef}
          onClick={handleAskFromHighlight}
        />
      )}

      {/* Generate audio: first scene via higgs-tts */}
      {generateModalOpen && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-[2px]"
          role="dialog"
          aria-modal="true"
          aria-labelledby="generate-audio-title"
        >
          <button
            type="button"
            className="absolute inset-0"
            onClick={closeGenerateModal}
            aria-label="Close"
          />
          <div
            className="relative mx-4 w-full max-w-md rounded-xl border border-border bg-background p-5 shadow-xl"
            onClick={(e) => e.stopPropagation()}
          >
            <h2 id="generate-audio-title" className="font-mono text-sm font-bold text-foreground">
              Generate audio — first scene
            </h2>
            <p className="mt-2 font-mono text-xs text-muted-foreground">
              Generates speech for the first scene via higgs-tts. Audio will play when ready.
            </p>
            {generateError && (
              <p className="mt-3 font-mono text-xs text-destructive">{generateError}</p>
            )}
            <div className="mt-5 flex justify-end gap-2">
              <button
                type="button"
                onClick={closeGenerateModal}
                disabled={audioGenerating}
                className={cn(
                  "rounded-lg border border-border px-3 py-1.5 font-mono text-xs",
                  "text-muted-foreground hover:bg-accent hover:text-accent-foreground disabled:opacity-50",
                )}
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={handleApproveAudioGeneration}
                disabled={audioGenerating}
                className={cn(
                  "rounded-lg border border-primary bg-primary px-3 py-1.5 font-mono text-xs text-primary-foreground",
                  "hover:bg-primary/90 disabled:opacity-50",
                )}
              >
                {audioGenerating ? "Generating…" : "Generate"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
