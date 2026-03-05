"use client";

import { useCallback } from "react";
import { useRouter } from "next/navigation";
import ScreenplayViewer from "@/components/ScreenplayViewer";
import type { Screenplay } from "@/types/screenplay";

export default function ScreenplayPageClient({
  screenplay,
  screenplayId,
}: {
  screenplay: Screenplay;
  screenplayId: string;
}) {
  const router = useRouter();
  const handleReset = useCallback(() => router.push("/"), [router]);

  return (
    <ScreenplayViewer
      screenplay={screenplay}
      screenplayId={screenplayId}
      onReset={handleReset}
    />
  );
}
