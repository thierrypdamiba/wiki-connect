import { cacheLinkupResult, searchLinkupCache } from "./qdrant";

const LINKUP_API_KEY = process.env.LINKUP_API_KEY;
const LINKUP_API_URL = "https://api.linkup.so/v1/search";

export interface LinkupResult {
  url: string;
  title: string;
  snippet: string;
  content: string;
}

export function isAvailable(): boolean {
  return !!LINKUP_API_KEY;
}

export async function searchWeb(
  query: string,
  depth: "standard" | "deep" = "standard"
): Promise<LinkupResult[]> {
  if (!LINKUP_API_KEY) {
    return [];
  }

  try {
    const response = await fetch(LINKUP_API_URL, {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${LINKUP_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        q: query,
        depth,
        outputType: "searchResults",
      }),
    });

    if (!response.ok) {
      console.error("Linkup API error:", response.status);
      return [];
    }

    const data = await response.json();

    if (data.results && Array.isArray(data.results)) {
      return data.results.map((r: { url?: string; name?: string; content?: string }) => ({
        url: r.url || "",
        title: r.name || "",
        snippet: (r.content || "").slice(0, 500),
        content: r.content || "",
      }));
    }

    return [];
  } catch (e) {
    console.error("Linkup search error:", e);
    return [];
  }
}

export async function searchAndCache(query: string, deep: boolean = false): Promise<LinkupResult[]> {
  // Check cache first
  const cached = await searchLinkupCache(query, 3);
  if (cached.length >= 2) {
    return cached;
  }

  // Fetch fresh results
  const results = await searchWeb(query, deep ? "deep" : "standard");

  // Cache the results
  for (const r of results.slice(0, 5)) {
    await cacheLinkupResult(query, r.url, r.title, r.snippet, r.content, 24);
  }

  return results;
}

export function formatForGrounding(results: LinkupResult[]): string {
  if (!results.length) {
    return "No web results found.";
  }

  const lines = ["**Web Search Results (Linkup):**\n"];

  for (let i = 0; i < Math.min(results.length, 5); i++) {
    const r = results[i];
    lines.push(`${i + 1}. **${r.title}**`);
    lines.push(`   ${r.snippet.slice(0, 300)}...`);
    lines.push(`   Source: ${r.url}\n`);
  }

  return lines.join("\n");
}
