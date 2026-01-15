import {
  CopilotRuntime,
  copilotRuntimeNextJSAppRouterEndpoint,
  ExperimentalEmptyAdapter,
} from "@copilotkit/runtime";
import { NextRequest } from "next/server";

// Backend URL for the ADK agent
const ADK_BACKEND_URL = process.env.ADK_BACKEND_URL || "http://localhost:9843";

const runtime = new CopilotRuntime({
  actions: [
    {
      name: "connectTopics",
      description: "Connect two Wikipedia topics and generate a synthesized article",
      parameters: [
        {
          name: "topicA",
          type: "string",
          description: "First topic to connect",
          required: true,
        },
        {
          name: "topicB",
          type: "string",
          description: "Second topic to connect",
          required: true,
        },
      ],
      handler: async ({ topicA, topicB }: { topicA: string; topicB: string }) => {
        try {
          const response = await fetch(`${ADK_BACKEND_URL}/connect`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ topic_a: topicA, topic_b: topicB }),
          });
          const data = await response.json();
          return JSON.stringify(data);
        } catch (error) {
          return JSON.stringify({ error: "Failed to connect topics" });
        }
      },
    },
    {
      name: "searchWikipedia",
      description: "Search Wikipedia for factual information on a topic",
      parameters: [
        {
          name: "query",
          type: "string",
          description: "The topic to search for",
          required: true,
        },
      ],
      handler: async ({ query }: { query: string }) => {
        try {
          const response = await fetch(`${ADK_BACKEND_URL}/search?query=${encodeURIComponent(query)}`);
          const data = await response.json();
          return JSON.stringify(data);
        } catch (error) {
          return JSON.stringify({ error: "Failed to search Wikipedia" });
        }
      },
    },
    {
      name: "generateImage",
      description: "Generate an illustration for a topic connection",
      parameters: [
        {
          name: "topicA",
          type: "string",
          description: "First topic",
          required: true,
        },
        {
          name: "topicB",
          type: "string",
          description: "Second topic",
          required: true,
        },
        {
          name: "connectionSummary",
          type: "string",
          description: "Brief description of how the topics connect",
          required: true,
        },
      ],
      handler: async ({
        topicA,
        topicB,
        connectionSummary,
      }: {
        topicA: string;
        topicB: string;
        connectionSummary: string;
      }) => {
        try {
          const response = await fetch(`${ADK_BACKEND_URL}/generate-image`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              topic_a: topicA,
              topic_b: topicB,
              connection_summary: connectionSummary,
            }),
          });
          const data = await response.json();
          return JSON.stringify(data);
        } catch (error) {
          return JSON.stringify({ error: "Failed to generate image" });
        }
      },
    },
    {
      name: "submitFeedback",
      description: "Submit feedback on a generated article",
      parameters: [
        {
          name: "articleId",
          type: "string",
          description: "ID of the article",
          required: true,
        },
        {
          name: "feedbackType",
          type: "string",
          description: "Type: text_style, image_style, format, or content",
          required: true,
        },
        {
          name: "feedbackText",
          type: "string",
          description: "The feedback content",
          required: true,
        },
        {
          name: "rating",
          type: "number",
          description: "Rating from 1-5",
          required: true,
        },
      ],
      handler: async ({
        articleId,
        feedbackType,
        feedbackText,
        rating,
      }: {
        articleId: string;
        feedbackType: string;
        feedbackText: string;
        rating: number;
      }) => {
        try {
          const response = await fetch(`${ADK_BACKEND_URL}/feedback`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              article_id: articleId,
              feedback_type: feedbackType,
              feedback_text: feedbackText,
              rating,
            }),
          });
          const data = await response.json();
          return JSON.stringify(data);
        } catch (error) {
          return JSON.stringify({ error: "Failed to submit feedback" });
        }
      },
    },
  ],
});

export const POST = async (req: NextRequest) => {
  const { handleRequest } = copilotRuntimeNextJSAppRouterEndpoint({
    runtime,
    serviceAdapter: new ExperimentalEmptyAdapter(),
    endpoint: "/api/copilotkit",
  });

  return handleRequest(req);
};
