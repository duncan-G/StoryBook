"use client";

import { cn } from "@/lib/utils";
import { SceneElement } from "@/types/screenplay";
import GhostInput from "./GhostInput";

interface Props {
  element: SceneElement;
  activeReferenceText?: string | null;
  onTextChange?: (newText: string) => void;
  /** Called when the user starts editing this element; use to clear the active reference highlight. */
  onClearActiveReference?: () => void;
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

export default function ElementRenderer({
  element,
  activeReferenceText,
  onTextChange,
  onClearActiveReference,
}: Props) {
  const active = isActiveElement(element.text, activeReferenceText);
  const activeAttr = active ? "" : undefined;

  // Highlight layer: absolutely positioned so it doesn't affect layout (no text shift).
  // Extends 12px around content for visual breathing room.
  const highlightLayer = active ? (
    <span
      aria-hidden
      className="pointer-events-none absolute -inset-3 rounded-sm border border-primary/50 bg-primary/5 shadow-[0_0_12px_rgba(217,119,6,0.15)] dark:shadow-[0_0_12px_rgba(245,158,11,0.2)]"
    />
  ) : null;
  // Always wrap in a stable container so clearing the highlight doesn't change the tree
  // and remount GhostInput (which would cancel entering edit mode on first click).
  const contentWrap = (children: React.ReactNode) => (
    <div className={active ? "relative z-10" : undefined}>{children}</div>
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
