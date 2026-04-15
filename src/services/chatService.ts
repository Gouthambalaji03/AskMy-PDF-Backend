/**
 * chatService.ts — LLM Chat with Streaming, Conversational Memory & Fallback
 *
 * Features:
 *   - Streaming responses via SSE (Server-Sent Events)
 *   - Conversational memory (last N Q&A pairs sent as context)
 *   - Retry with exponential backoff on 429
 *   - Model fallback: gemini-2.5-flash → gemini-2.0-flash
 *   - Metadata-aware prompts with page/section citations
 */

import { GoogleGenerativeAI, GenerativeModel } from "@google/generative-ai";
import { SearchResult } from "../store/vectorStore";

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY!);

const PRIMARY_MODEL = "gemini-2.5-flash";
const FALLBACK_MODEL = "gemini-2.0-flash";

const chatModel = genAI.getGenerativeModel({
  model: PRIMARY_MODEL,
  generationConfig: {
    temperature: 0.3,
    maxOutputTokens: 1000,
  },
});

const fallbackModel = genAI.getGenerativeModel({
  model: FALLBACK_MODEL,
  generationConfig: {
    temperature: 0.3,
    maxOutputTokens: 1000,
  },
});

// ── Conversation History Type ─────────────────────────────────

export interface ChatTurn {
  question: string;
  answer: string;
}

// ── Prompt Builder ────────────────────────────────────────────

function buildPrompt(
  question: string,
  contextChunks: SearchResult[],
  chatHistory: ChatTurn[] = []
): string {
  const contextText = contextChunks
    .map((chunk, index) => {
      const page = chunk.metadata?.pageNumber
        ? `Page ${chunk.metadata.pageNumber}`
        : "Unknown page";
      const section = chunk.metadata?.sectionHeading
        ? ` | Section: ${chunk.metadata.sectionHeading}`
        : "";
      const matchInfo = chunk.matchType
        ? ` | Found via: ${chunk.matchType} search`
        : "";
      return `[Source ${index + 1} — ${page}${section}${matchInfo}]:\n${chunk.text}`;
    })
    .join("\n\n");

  // Build conversation history section
  let historySection = "";
  if (chatHistory.length > 0) {
    historySection = "\n\nPrevious conversation:\n" +
      chatHistory.map((turn) =>
        `User: ${turn.question}\nAssistant: ${turn.answer}`
      ).join("\n\n") +
      "\n";
  }

  const systemPrompt = `You are a helpful assistant that answers questions based ONLY on the provided context from a PDF document.

Rules:
- Only use information from the context below to answer the question.
- If the context doesn't contain enough information to answer, say "I don't have enough information from the PDF to answer this question."
- Do NOT make up or infer information beyond what's explicitly stated in the context.
- Keep your answers concise and well-structured.
- When referencing information, cite the page number and section (e.g., "According to Page 5, Section 2.1...").
- If multiple sources support your answer, mention all relevant page numbers.
- Use the previous conversation (if any) to understand follow-up questions. Resolve pronouns like "it", "this", "that" using prior context.

Context from the PDF:
${contextText}${historySection}`;

  return `${systemPrompt}\n\nUser question: ${question}`;
}

// ── Retry Logic ───────────────────────────────────────────────

async function callWithRetry(
  model: GenerativeModel,
  prompt: string,
  maxRetries: number = 3
): Promise<string> {
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      const result = await model.generateContent(prompt);
      return result.response.text() || "Sorry, I could not generate an answer.";
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "";
      const is429 = msg.includes("429") || msg.includes("Too Many Requests") || msg.includes("RESOURCE_EXHAUSTED");

      if (is429 && attempt < maxRetries) {
        const waitMs = Math.pow(2, attempt + 1) * 1000;
        console.log(`⏳ Rate limited, retrying in ${waitMs / 1000}s (attempt ${attempt + 1}/${maxRetries})...`);
        await new Promise((resolve) => setTimeout(resolve, waitMs));
        continue;
      }
      throw err;
    }
  }
  throw new Error("Max retries exceeded");
}

// ── Non-Streaming (kept for cache hits) ───────────────────────

export async function askQuestion(
  question: string,
  contextChunks: SearchResult[],
  chatHistory: ChatTurn[] = []
): Promise<string> {
  const fullPrompt = buildPrompt(question, contextChunks, chatHistory);

  try {
    return await callWithRetry(chatModel, fullPrompt);
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : "";

    if (msg.includes("503") || msg.includes("Service Unavailable") || msg.includes("high demand")) {
      console.log(`⚠️  ${PRIMARY_MODEL} unavailable, falling back to ${FALLBACK_MODEL}`);
      try {
        return await callWithRetry(fallbackModel, fullPrompt);
      } catch (fallbackErr: unknown) {
        const fbMsg = fallbackErr instanceof Error ? fallbackErr.message : "";
        if (fbMsg.includes("429") || fbMsg.includes("quota")) throw new Error("QUOTA_EXHAUSTED");
        throw fallbackErr;
      }
    }

    if (msg.includes("429") || msg.includes("Too Many Requests") || msg.includes("RESOURCE_EXHAUSTED")) {
      console.log(`⚠️  ${PRIMARY_MODEL} rate limited after retries, trying ${FALLBACK_MODEL}`);
      try {
        return await callWithRetry(fallbackModel, fullPrompt);
      } catch (fallbackErr: unknown) {
        const fbMsg = fallbackErr instanceof Error ? fallbackErr.message : "";
        if (fbMsg.includes("429") || fbMsg.includes("quota")) throw new Error("QUOTA_EXHAUSTED");
        throw fallbackErr;
      }
    }

    throw err;
  }
}

// ── Streaming Response ────────────────────────────────────────

/**
 * Streams the LLM response chunk-by-chunk via a callback.
 * Falls back to the secondary model on 503/429.
 * Returns the full assembled answer when done.
 */
export async function askQuestionStream(
  question: string,
  contextChunks: SearchResult[],
  chatHistory: ChatTurn[] = [],
  onChunk: (text: string) => void
): Promise<string> {
  const fullPrompt = buildPrompt(question, contextChunks, chatHistory);

  async function streamFromModel(model: GenerativeModel): Promise<string> {
    const result = await model.generateContentStream(fullPrompt);
    let fullText = "";

    for await (const chunk of result.stream) {
      const text = chunk.text();
      if (text) {
        fullText += text;
        onChunk(text);
      }
    }

    return fullText || "Sorry, I could not generate an answer.";
  }

  // Try primary model
  try {
    return await streamFromModel(chatModel);
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : "";

    const shouldFallback =
      msg.includes("503") || msg.includes("Service Unavailable") || msg.includes("high demand") ||
      msg.includes("429") || msg.includes("Too Many Requests") || msg.includes("RESOURCE_EXHAUSTED");

    if (shouldFallback) {
      console.log(`⚠️  ${PRIMARY_MODEL} failed for streaming, falling back to ${FALLBACK_MODEL}`);
      try {
        return await streamFromModel(fallbackModel);
      } catch (fallbackErr: unknown) {
        const fbMsg = fallbackErr instanceof Error ? fallbackErr.message : "";
        if (fbMsg.includes("429") || fbMsg.includes("quota")) throw new Error("QUOTA_EXHAUSTED");
        throw fallbackErr;
      }
    }

    throw err;
  }
}
