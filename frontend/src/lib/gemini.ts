import { GoogleGenerativeAI } from "@google/generative-ai";

const GOOGLE_API_KEY = process.env.GOOGLE_API_KEY!;

let _client: GoogleGenerativeAI | null = null;

function getClient(): GoogleGenerativeAI {
  if (!_client) {
    _client = new GoogleGenerativeAI(GOOGLE_API_KEY);
  }
  return _client;
}

export async function generateContent(prompt: string): Promise<string> {
  const client = getClient();
  const model = client.getGenerativeModel({ model: "gemini-2.5-flash" });

  const result = await model.generateContent(prompt);
  return result.response.text();
}

export async function* generateContentStream(prompt: string): AsyncGenerator<string> {
  const client = getClient();
  const model = client.getGenerativeModel({ model: "gemini-2.5-flash" });

  const result = await model.generateContentStream(prompt);

  for await (const chunk of result.stream) {
    const text = chunk.text();
    if (text) {
      yield text;
    }
  }
}

export async function generateImage(prompt: string): Promise<{ imageData: string; mimeType: string } | null> {
  const client = getClient();

  try {
    const model = client.getGenerativeModel({ model: "gemini-2.5-flash" });

    const result = await model.generateContent({
      contents: [{ role: "user", parts: [{ text: prompt }] }],
    });

    const response = result.response;
    const candidates = response.candidates;

    if (candidates && candidates.length > 0) {
      const parts = candidates[0].content.parts;
      for (const part of parts) {
        if ("inlineData" in part && part.inlineData) {
          return {
            imageData: part.inlineData.data,
            mimeType: part.inlineData.mimeType,
          };
        }
      }
    }
  } catch (e) {
    console.error("Image generation error:", e);
  }

  return null;
}

export function buildArticlePrompt(
  topicA: string,
  topicB: string,
  contextA: string,
  contextB: string,
  contextConn: string,
  prefInstructions: string = ""
): string {
  return `You are a Wikipedia-style article writer. Create an engaging article that connects two topics.

**GROUNDING SOURCES (use ONLY these facts):**

About ${topicA}:
${contextA}

About ${topicB}:
${contextB}

Connection context:
${contextConn || "Find conceptual bridges between the topics."}
${prefInstructions}

**TASK:**
Write a wiki-style article (800-1200 words) explaining the fascinating connection between "${topicA}" and "${topicB}".

Requirements:
1. Use ONLY facts from the grounding sources above
2. Find creative but accurate connections
3. Use clear headers and structure
4. Cite sources inline: "According to Wikipedia's [Topic] article..."
5. Make it engaging and educational
${prefInstructions ? "6. IMPORTANT: Apply the user's preferences listed above to customize your writing style" : ""}

Output the article in Markdown format with a creative title.`;
}

export function buildImagePrompt(topicA: string, topicB: string, connectionSummary: string): string {
  return `Create an artistic, educational illustration showing the connection between "${topicA}" and "${topicB}".

The image should visually represent: ${connectionSummary}

Style: Clean, modern infographic style with vibrant colors. Scientific accuracy meets artistic beauty. No text or labels in the image. Abstract representation of concepts.`;
}
