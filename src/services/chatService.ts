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
 * The system prompt instructs the LLM to:
 *   - Reference page numbers when citing information
 *   - Mention section headings for navigation
 *   - Indicate the match type (semantic/keyword/hybrid)
 * =============================================================
 */

import { GoogleGenerativeAI } from "@google/generative-ai";
import { SearchResult } from "./embeddingService";

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
 * Sends the user's question + relevant context chunks to Gemini
 * and gets back an answer grounded in the provided context.
 *
 * Now includes page numbers and section headings in the context
 * so the LLM can provide citations in its answers.
 *
 * @param question - The user's question
 * @param contextChunks - The top relevant chunks from hybrid search + re-ranking
 * @returns The AI-generated answer
 */
export async function askQuestion(
  question: string,
  contextChunks: SearchResult[]
): Promise<string> {
  // Format context with metadata for the LLM
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
    const result = await chatModel.generateContent(fullPrompt);
    return result.response.text() || "Sorry, I could not generate an answer.";
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : "";
    if (msg.includes("503") || msg.includes("Service Unavailable") || msg.includes("high demand")) {
      console.log(`⚠️  ${PRIMARY_MODEL} unavailable, falling back to ${FALLBACK_MODEL}`);
      const result = await fallbackModel.generateContent(fullPrompt);
      return result.response.text() || "Sorry, I could not generate an answer.";
    }
    throw err;
  }
}
