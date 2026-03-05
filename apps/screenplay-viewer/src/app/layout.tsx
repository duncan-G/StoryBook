import type { Metadata } from "next";
import { Geist_Mono } from "next/font/google";
import { trace } from "@opentelemetry/api";
import ThemeProvider from "@/components/ThemeProvider";
import TelemetryProvider from "@/components/TelemetryProvider";
import "./globals.scss";

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

/**
 * Dynamic metadata so we can inject <meta name="traceparent"> from the active
 * server span. DocumentLoadInstrumentation reads this tag and parents the
 * browser's documentLoad span under the server trace — one unified trace.
 */
export async function generateMetadata(): Promise<Metadata> {
  const meta: Metadata = {
    title: "Screenplay Viewer",
    description: "Upload and read screenplays in a structured viewer.",
  };

  const span = trace.getActiveSpan();
  if (span) {
    const { traceId, spanId, traceFlags } = span.spanContext();
    const sampled = (traceFlags & 0x01) === 0x01 ? "01" : "00";
    meta.other = {
      traceparent: `00-${traceId}-${spanId}-${sampled}`,
    };
  }

  return meta;
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={`${geistMono.variable} antialiased bg-background text-foreground`}>
        <ThemeProvider>
          <TelemetryProvider>{children}</TelemetryProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}
