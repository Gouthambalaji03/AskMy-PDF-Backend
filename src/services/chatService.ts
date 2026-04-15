/**
 * chatService.ts — LLM Chat Completion with Context (Google Gemini)
 *
 * =============================================================
 * TIER 2: Metadata-Aware Prompt Engineering
 * =============================================================
 *
 * Updated to include chunk metadata (page number, section heading)
 * in the context sent to the LLM. This allows the AI to cite
 * specific pages and sections in its answers, making responses
 * more trustworthy and verifiable.
 *
 * Includes automatic retry with exponential backoff for rate
 * limits (429) and model fallback for service outages (503).
 * =============================================================
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

/**
 * Calls a Gemini model with automatic retry on 429 (rate limit).
 * Uses exponential backoff: waits 2s, then 4s, then 8s.
 * Gives up after maxRetries and throws the original error.
 */
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
        const waitMs = Math.pow(2, attempt + 1) * 1000; // 2s, 4s, 8s
        console.log(`⏳ Rate limited, retrying in ${waitMs / 1000}s (attempt ${attempt + 1}/${maxRetries})...`);
        await new Promise((resolve) => setTimeout(resolve, waitMs));
        continue;
      }
      throw err;
    }
  }
  throw new Error("Max retries exceeded");
}

/**
 * Sends the user's question + relevant context chunks to Gemini
 * and gets back an answer grounded in the provided context.
 *
 * Retry strategy:
 *   1. Try primary model (gemini-2.5-flash) with up to 3 retries
 *   2. If 503 (overloaded), fall back to gemini-2.0-flash with retries
 *   3. If daily quota is fully exhausted, return a clear error message
 */
export async function askQuestion(
  question: string,
  contextChunks: SearchResult[]
): Promise<string> {
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

  const systemPrompt = `You are a helpful assistant that answers questions based ONLY on the provided context from a PDF document.

Rules:
- Only use information from the context below to answer the question.
- If the context doesn't contain enough information to answer, say "I don't have enough information from the PDF to answer this question."
- Do NOT make up or infer information beyond what's explicitly stated in the context.
- Keep your answers concise and well-structured.
- When referencing information, cite the page number and section (e.g., "According to Page 5, Section 2.1...").
- If multiple sources support your answer, mention all relevant page numbers.

Context from the PDF:
${contextText}`;

  const fullPrompt = `${systemPrompt}\n\nUser question: ${question}`;

  try {
    return await callWithRetry(chatModel, fullPrompt);
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : "";

    // 503: model overloaded → try fallback model
    if (msg.includes("503") || msg.includes("Service Unavailable") || msg.includes("high demand")) {
      console.log(`⚠️  ${PRIMARY_MODEL} unavailable, falling back to ${FALLBACK_MODEL}`);
      try {
        return await callWithRetry(fallbackModel, fullPrompt);
      } catch (fallbackErr: unknown) {
        const fbMsg = fallbackErr instanceof Error ? fallbackErr.message : "";
        if (fbMsg.includes("429") || fbMsg.includes("quota")) {
          throw new Error("QUOTA_EXHAUSTED");
        }
        throw fallbackErr;
      }
    }

    // 429 on primary after all retries → try fallback
    if (msg.includes("429") || msg.includes("Too Many Requests") || msg.includes("RESOURCE_EXHAUSTED")) {
      console.log(`⚠️  ${PRIMARY_MODEL} rate limited after retries, trying ${FALLBACK_MODEL}`);
      try {
        return await callWithRetry(fallbackModel, fullPrompt);
      } catch (fallbackErr: unknown) {
        const fbMsg = fallbackErr instanceof Error ? fallbackErr.message : "";
        if (fbMsg.includes("429") || fbMsg.includes("quota")) {
          throw new Error("QUOTA_EXHAUSTED");
        }
        throw fallbackErr;
      }
    }

    throw err;
  }
}
