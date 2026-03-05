"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { FileText, Trash2 } from "lucide-react";
import { ScreenplayListItem } from "@/types/screenplay";
import { cn } from "@/lib/utils";

interface Props {
  items: ScreenplayListItem[];
  onSelect: (id: string) => void;
  onDelete?: (id: string) => void;
  selectedId?: string | null;
  loading?: boolean;
}

export default function ScreenplayList({
  items,
  onSelect,
  onDelete,
  selectedId,
  loading,
}: Props) {
  const [contextMenu, setContextMenu] = useState<{
    x: number;
    y: number;
    itemId: string;
  } | null>(null);
  const menuRef = useRef<HTMLDivElement>(null);

  const closeMenu = useCallback(() => setContextMenu(null), []);

  useEffect(() => {
    if (!contextMenu) return;
    const handleClickOutside = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        closeMenu();
      }
    };
    const handleScroll = () => closeMenu();
    document.addEventListener("click", handleClickOutside);
    document.addEventListener("scroll", handleScroll, true);
    return () => {
      document.removeEventListener("click", handleClickOutside);
      document.removeEventListener("scroll", handleScroll, true);
    };
  }, [contextMenu, closeMenu]);

  const handleContextMenu = useCallback(
    (e: React.MouseEvent, itemId: string) => {
      e.preventDefault();
      if (onDelete) {
        setContextMenu({ x: e.clientX, y: e.clientY, itemId });
      }
    },
    [onDelete],
  );

  const handleDelete = useCallback(
    (itemId: string) => {
      onDelete?.(itemId);
      closeMenu();
    },
    [onDelete, closeMenu],
  );

  if (loading) {
    return (
      <div className="flex min-h-[200px] items-center justify-center">
        <div className="h-8 w-8 animate-spin rounded-full border-4 border-muted border-t-primary" />
      </div>
    );
  }

  if (items.length === 0) {
    return null;
  }

  return (
    <>
      <ul className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
        {items.map((item) => (
          <li key={item.id}>
            <button
              type="button"
              onClick={() => onSelect(item.id)}
              onContextMenu={(e) => handleContextMenu(e, item.id)}
              className={cn(
                "flex w-full items-start gap-3 rounded-xl border px-4 py-3 text-left transition-colors",
                "border-border bg-card hover:border-primary/50 hover:bg-accent/50",
                selectedId === item.id && "border-primary bg-accent/50",
              )}
            >
              <FileText className="mt-0.5 h-5 w-5 shrink-0 text-muted-foreground" />
              <div className="min-w-0 flex-1">
                <div className="truncate font-mono text-sm font-medium text-foreground">
                  {item.title || "Untitled"}
                </div>
                {item.authors.length > 0 && (
                  <div className="truncate font-mono text-xs text-muted-foreground">
                    {item.authors.join(", ")}
                  </div>
                )}
                {item.source_filename && (
                  <div className="mt-1 truncate font-mono text-xs text-muted-foreground/70">
                    {item.source_filename}
                  </div>
                )}
              </div>
            </button>
          </li>
        ))}
      </ul>

      {contextMenu && (
        <div
          ref={menuRef}
          className="fixed z-50 min-w-[140px] rounded-lg border border-border bg-card py-1 shadow-lg"
          style={{ left: contextMenu.x, top: contextMenu.y }}
        >
          <button
            type="button"
            onClick={() => handleDelete(contextMenu.itemId)}
            className="flex w-full items-center gap-2 px-3 py-2 font-mono text-sm text-foreground hover:bg-destructive/10 hover:text-destructive"
          >
            <Trash2 className="h-4 w-4" />
            Delete
          </button>
        </div>
      )}
    </>
  );
}
