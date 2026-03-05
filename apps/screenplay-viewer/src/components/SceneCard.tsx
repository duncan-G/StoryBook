"use client";

import { useCallback, useEffect, useState } from "react";
import { cn } from "@/lib/utils";
import { updateElementText, updateSceneMetadata } from "@/lib/api";
import { Scene } from "@/types/screenplay";
import ElementRenderer from "./ElementRenderer";
import GhostInput from "./GhostInput";
import GhostSelect from "./GhostSelect";

const LOCATION_TYPES = ["—", "INT", "EXT", "I/E"];
const COMMON_TIMES = ["—", "DAY", "NIGHT", "DAWN", "DUSK", "MORNING", "EVENING", "CONTINUOUS", "LATER", "MOMENTS LATER"];

interface Props {
  scene: Scene;
  index: number;
  total: number;
  activeReferenceText?: string | null;
  onClearActiveReference?: () => void;
  screenplayId?: string;
}

export default function SceneCard({
  scene,
  index,
  total,
  activeReferenceText,
  onClearActiveReference,
  screenplayId,
}: Props) {
  const [locType, setLocType] = useState(scene.location_type ?? "");
  const [location, setLocation] = useState(scene.location || "");
  const [timeOfDay, setTimeOfDay] = useState(scene.time_of_day || "");
  const [elements, setElements] = useState(scene.elements);

  // Sync when the scene prop changes (e.g. navigating to a different scene)
  useEffect(() => {
    setLocType(scene.location_type ?? "");
    setLocation(scene.location || "");
    setTimeOfDay(scene.time_of_day || "");
    setElements(scene.elements);
  }, [scene]);

  const patchScene = useCallback(
    (patch: { location_type?: string; location?: string; time_of_day?: string }) => {
      if (!screenplayId) return;
      updateSceneMetadata(screenplayId, index, patch).catch(() => {
        setLocType(scene.location_type ?? "");
        setLocation(scene.location || "");
        setTimeOfDay(scene.time_of_day || "");
      });
    },
    [screenplayId, index, scene],
  );

  const commitLocType = useCallback(
    (val: string) => {
      const normalized = val === "—" ? "" : val;
      setLocType(normalized);
      patchScene({ location_type: normalized });
    },
    [patchScene],
  );

  const commitLocation = useCallback(
    (val: string) => {
      setLocation(val);
      patchScene({ location: val });
    },
    [patchScene],
  );

  const commitTime = useCallback(
    (val: string) => {
      const normalized = val === "—" ? "" : val;
      setTimeOfDay(normalized);
      patchScene({ time_of_day: normalized });
    },
    [patchScene],
  );

  const handleElementTextChange = useCallback(
    (elementIndex: number) => (newText: string) => {
      setElements((prev) => {
        const next = [...prev];
        next[elementIndex] = { ...next[elementIndex], text: newText };
        return next;
      });
      if (!screenplayId) return;
      updateElementText(screenplayId, index, elementIndex, newText).catch(() => {
        setElements(scene.elements);
      });
    },
    [screenplayId, index, scene.elements],
  );

  return (
    <article className="w-full">
      <header
        className={cn(
          "sticky top-0 z-10 flex flex-wrap items-center gap-2 px-3 py-2 sm:gap-3 sm:px-6 sm:py-3",
          "border-b border-border bg-background/90 backdrop-blur-sm",
        )}
      >
        <span
          className={cn(
            "rounded px-2 py-0.5 font-mono text-xs font-semibold",
            "bg-scene-badge-bg text-scene-badge-text",
          )}
        >
          {index + 1}/{total}
        </span>

        {/* Location type: dropdown */}
        <GhostSelect
          value={locType || "—"}
          options={LOCATION_TYPES}
          placeholder="—"
          onCommit={commitLocType}
          triggerClassName="bg-loc-badge-bg text-loc-badge-text"
        />

        {/* Location: editable bold text */}
        <div className="order-2 min-w-0 flex-1 sm:order-none">
          <GhostInput
            value={location}
            placeholder="Location"
            onCommit={commitLocation}
            className="font-mono text-sm font-semibold uppercase text-foreground"
          />
        </div>

        {/* Time of day: dropdown + custom */}
        <GhostSelect
          value={timeOfDay || "—"}
          options={COMMON_TIMES}
          placeholder="—"
          allowCustom
          customPlaceholder="Custom time…"
          onCommit={commitTime}
          triggerClassName="text-muted-foreground"
          align="right"
        />

        <span className="font-mono text-xs text-muted-foreground/60">p.{scene.page}</span>
      </header>

      {scene.characters_present.length > 0 && (
        <div className="flex flex-wrap gap-1.5 border-b border-border/50 px-3 py-2 sm:px-6">
          {scene.characters_present.map((c) => (
            <span
              key={c}
              className={cn(
                "rounded-full px-2.5 py-0.5 font-mono text-[11px]",
                "bg-char-pill-bg text-char-pill-text",
              )}
            >
              {c}
            </span>
          ))}
        </div>
      )}

      <div className="flex flex-col items-center px-3 py-3 sm:px-6 sm:py-4">
        {elements.map((el, i) => {
          if (el.character && (el.type === "DIALOGUE" || el.type === "VOICE_OVER")) {
            const isFirstInRun =
              i === 0 ||
              elements[i - 1].character !== el.character ||
              (elements[i - 1].type !== "DIALOGUE" &&
                elements[i - 1].type !== "VOICE_OVER" &&
                elements[i - 1].type !== "PARENTHETICAL");

            return (
              <div key={i} className="w-full max-w-[540px]">
                {isFirstInRun && (
                  <div className="mb-1 mt-4 ml-10 font-mono text-xs font-bold uppercase tracking-widest text-char-cue sm:ml-20">
                    {el.character}
                    {el.type === "VOICE_OVER" && (
                      <span className="ml-1 text-vo-text/70">(V.O.)</span>
                    )}
                  </div>
                )}
                <ElementRenderer
                  element={el}
                  activeReferenceText={activeReferenceText}
                  onTextChange={screenplayId ? handleElementTextChange(i) : undefined}
                  onClearActiveReference={onClearActiveReference}
                />
              </div>
            );
          }
          return (
            <ElementRenderer
              key={i}
              element={el}
              activeReferenceText={activeReferenceText}
              onTextChange={screenplayId ? handleElementTextChange(i) : undefined}
              onClearActiveReference={onClearActiveReference}
            />
          );
        })}
      </div>
    </article>
  );
}
