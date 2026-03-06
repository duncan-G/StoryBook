"use client";

import { Volume2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { SceneElement } from "@/types/screenplay";
import GhostInput from "./GhostInput";

interface Props {
  element: SceneElement;
  activeReferenceText?: string | null;
  onTextChange?: (newText: string) => void;
  /** Called when the user starts editing this element; use to clear the active reference highlight. */
  onClearActiveReference?: () => void;
  /** Called when the generate-audio button is clicked. Triggers scene-level audio generation. */
  onGenerateAudio?: () => void;
  /** When true, the generate-audio button is disabled (e.g. generation in progress). */
  audioLoading?: boolean;
  /** When true, show the generate-audio button when the input is focused. */
  showGenerateAudio?: boolean;
}

const base = "font-mono text-sm";

function isActiveElement(
  elementText: string,
  activeReferenceText: string | null | undefined,
): boolean {
  if (!activeReferenceText?.trim()) return false;
  const a = elementText.trim();
  const b = activeReferenceText.trim();
  return (
    a === b ||
    a.includes(b) ||
    b.includes(a)
  );
}

function EditableOrStatic({
  text,
  onTextChange,
  onEditingStart,
  className,
  block,
}: {
  text: string;
  onTextChange?: (newText: string) => void;
  onEditingStart?: () => void;
  className?: string;
  block?: boolean;
}) {
  if (onTextChange) {
    return (
      <GhostInput
        value={text}
        placeholder=""
        onCommit={onTextChange}
        onEditingStart={onEditingStart}
        className={cn(className, block && "block w-full")}
        as={block ? "div" : "span"}
      />
    );
  }
  return <>{text}</>;
}

const audioButton = (
  onGenerateAudio: () => void,
  audioLoading: boolean,
) => (
  <button
    type="button"
    onClick={(e) => {
      e.stopPropagation();
      onGenerateAudio();
    }}
    disabled={audioLoading}
    aria-label={audioLoading ? "Generating audio…" : "Generate audio for this scene"}
    className={cn(
      "absolute -top-0.5 right-0 z-20 flex h-6 w-6 items-center justify-center rounded",
      "border border-border bg-background shadow-sm text-muted-foreground",
      "opacity-0 transition-opacity duration-150 focus-within:opacity-100 group-focus-within/input:opacity-100",
      "hover:bg-accent hover:text-accent-foreground",
      "focus:outline-none focus:ring-1 focus:ring-ring",
      "disabled:pointer-events-none disabled:opacity-40",
    )}
  >
    <Volume2 className="h-3 w-3" aria-hidden />
  </button>
);

export default function ElementRenderer({
  element,
  activeReferenceText,
  onTextChange,
  onClearActiveReference,
  onGenerateAudio,
  audioLoading = false,
  showGenerateAudio = false,
}: Props) {
  const active = isActiveElement(element.text, activeReferenceText);
  const activeAttr = active ? "" : undefined;

  // Highlight layer: absolutely positioned so it doesn't affect layout (no text shift).
  const highlightLayer = active ? (
    <span
      aria-hidden
      className="pointer-events-none absolute -inset-3 rounded-sm border border-primary/50 bg-primary/5 shadow-[0_0_12px_rgba(217,119,6,0.15)] dark:shadow-[0_0_12px_rgba(245,158,11,0.2)]"
    />
  ) : null;
  // Wrap content in group/input so focus-within reveals the audio button when the input is focused.
  const contentWrap = (children: React.ReactNode) => (
    <div
      className={cn(
        "relative group/input pr-8",
        active && "z-10",
      )}
    >
      {children}
      {showGenerateAudio && onGenerateAudio && onTextChange && audioButton(onGenerateAudio, audioLoading)}
    </div>
  );

  switch (element.type) {
    case "SCENE_HEADING":
      return (
        <div
          data-active-ref={activeAttr}
          className={cn(
            base,
            "my-6 font-bold uppercase tracking-wide text-scene-heading",
            active && "relative",
          )}
        >
          {highlightLayer}
          {contentWrap(
            <EditableOrStatic
              text={element.text}
              onTextChange={onTextChange}
              onEditingStart={active ? onClearActiveReference : undefined}
              className="font-bold uppercase tracking-wide text-scene-heading"
              block
            />,
          )}
        </div>
      );

    case "ACTION":
      return (
        <div
          data-active-ref={activeAttr}
          className={cn(
            base,
            "my-3 max-w-[540px] leading-relaxed text-action-text",
            active && "relative",
          )}
        >
          {highlightLayer}
          {contentWrap(
            <EditableOrStatic
              text={element.text}
              onTextChange={onTextChange}
              onEditingStart={active ? onClearActiveReference : undefined}
              className="leading-relaxed text-action-text"
              block
            />,
          )}
        </div>
      );

    case "DIALOGUE":
      return (
        <div
          data-active-ref={activeAttr}
          className={cn(
            base,
            "my-1 ml-8 max-w-[320px] leading-relaxed text-dialogue-text sm:ml-16 sm:max-w-[340px]",
            active && "relative",
          )}
        >
          {highlightLayer}
          {contentWrap(
            <EditableOrStatic
              text={element.text}
              onTextChange={onTextChange}
              onEditingStart={active ? onClearActiveReference : undefined}
              className="leading-relaxed text-dialogue-text"
              block
            />,
          )}
        </div>
      );

    case "VOICE_OVER":
      return (
        <div
          data-active-ref={activeAttr}
          className={cn(
            base,
            "my-1 ml-8 max-w-[320px] italic leading-relaxed text-vo-text sm:ml-16 sm:max-w-[340px]",
            active && "relative",
          )}
        >
          {highlightLayer}
          {contentWrap(
            <EditableOrStatic
              text={element.text}
              onTextChange={onTextChange}
              onEditingStart={active ? onClearActiveReference : undefined}
              className="italic leading-relaxed text-vo-text"
              block
            />,
          )}
        </div>
      );

    case "PARENTHETICAL":
      return (
        <div
          data-active-ref={activeAttr}
          className={cn(
            base,
            "my-0.5 ml-12 max-w-[220px] text-paren-text sm:ml-24 sm:max-w-[260px]",
            active && "relative",
          )}
        >
          {highlightLayer}
          {contentWrap(
            <EditableOrStatic
              text={element.text}
              onTextChange={onTextChange}
              onEditingStart={active ? onClearActiveReference : undefined}
              className="text-paren-text"
              block
            />,
          )}
        </div>
      );

    case "TRANSITION":
      return (
        <div
          data-active-ref={activeAttr}
          className={cn(
            base,
            "my-4 text-right font-bold uppercase text-transition-text",
            active && "relative",
          )}
        >
          {highlightLayer}
          {contentWrap(
            <EditableOrStatic
              text={element.text}
              onTextChange={onTextChange}
              onEditingStart={active ? onClearActiveReference : undefined}
              className="font-bold uppercase text-transition-text"
            />,
          )}
        </div>
      );

    case "SECTION_HEADER":
      return (
        <div
          data-active-ref={activeAttr}
          className={cn(
            base,
            "my-6 text-center font-bold uppercase tracking-widest text-muted-foreground",
            active && "relative",
          )}
        >
          {highlightLayer}
          {contentWrap(
            <EditableOrStatic
              text={element.text}
              onTextChange={onTextChange}
              onEditingStart={active ? onClearActiveReference : undefined}
              className="font-bold uppercase tracking-widest text-muted-foreground"
            />,
          )}
        </div>
      );

    default:
      return (
        <div
          data-active-ref={activeAttr}
          className={cn(base, "my-2 text-muted-foreground", active && "relative")}
        >
          {highlightLayer}
          {contentWrap(
            <EditableOrStatic
              text={element.text}
              onTextChange={onTextChange}
              onEditingStart={active ? onClearActiveReference : undefined}
              className="text-muted-foreground"
            />,
          )}
        </div>
      );
  }
}
