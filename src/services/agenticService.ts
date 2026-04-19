/**
 * agenticService.ts — Corrective RAG (CRAG) Self-Correction Layer
 *
 * =============================================================
 * CRAG: Corrective Retrieval-Augmented Generation
 * =============================================================
 *
 * Standard RAG retrieves chunks and trusts them blindly. If the
 * retrieved chunks are weak (off-topic, tangential, or missing
 * key info), the LLM either hallucinates or gives a weak answer.
 *
 * CRAG adds a self-check loop:
 *   1. Retrieve chunks (existing hybrid search)
 *   2. Grade whether the chunks can actually answer the question
 *   3. If low confidence → rewrite the query and retrieve again
 *   4. If still low → honestly report "not enough info"
 *
 * This module provides the two LLM-powered primitives:
 *   - gradeRelevance: judges whether retrieved chunks are useful
 *   - rewriteQuery: produces an improved query when retrieval fails
 *
 * Both use a fast/cheap model (gemini-2.0-flash) and degrade
 * gracefully if the grader/rewriter itself fails.
 * =============================================================
 */

import { GoogleGenerativeAI } from "@google/generative-ai";
import { SearchResult } from "../store/vectorStore";

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY!);

// Use the faster/cheaper model for agentic sub-tasks.
// These calls should be sub-second and not add meaningful latency.
const graderModel = genAI.getGenerativeModel({
  model: "gemini-2.5-flash",
  generationConfig: {
    temperature: 0.1, // deterministic grading
    maxOutputTokens: 512,
  },
});

const rewriterModel = genAI.getGenerativeModel({
  model: "gemini-2.5-flash",
  generationConfig: {
    temperature: 0.3,
    maxOutputTokens: 100,
  },
});

// ── Types ──────────────────────────────────────────────────────

export interface RelevanceGrade {
  confidence: "high" | "low";
  reason: string;
}

export interface CRAGInfo {
  triggered: boolean;                     // did we run the rewrite+retry?
  initialConfidence: "high" | "low";
  finalConfidence: "high" | "low";
  rewrittenQuery: string | null;          // what we rewrote to (null if not triggered)
  reason: string;                         // grader's latest reasoning
}

// ── Relevance Grader ──────────────────────────────────────────

/**
 * Grades whether the retrieved chunks contain enough information
 * to answer the user's question.
 *
 * Returns confidence "high" or "low". On any internal failure
 * (API error, invalid JSON), gracefully returns "high" so CRAG
 * acts as a no-op rather than blocking the answer.
 */
export async function gradeRelevance(
  question: string,
  chunks: SearchResult[]
): Promise<RelevanceGrade> {
  // No chunks at all → definitely low
  if (chunks.length === 0) {
    return { confidence: "low", reason: "No chunks retrieved." };
  }

  // Truncate chunks to keep prompt small & fast
  const chunkTexts = chunks
    .map((c, i) => `[${i + 1}]: ${c.text.slice(0, 400)}`)
    .join("\n\n");

  const prompt = `You are a retrieval relevance grader. Decide if the provided chunks contain enough information to answer the user's question.

Question: "${question}"

Retrieved chunks:
${chunkTexts}

Respond with ONLY a JSON object in this exact format (no prose, no markdown):
{"confidence": "high" | "low", "reason": "one short sentence under 15 words"}

Rules:
- "high" = chunks clearly contain information that addresses the question.
- "low"  = chunks are off-topic, tangential, or insufficient to answer.`;

  try {
    const result = await graderModel.generateContent(prompt);
    const raw = result.response.text().trim();

    // Gemini may wrap JSON in ```json ... ``` fences — strip them.
    // Also tolerate leading/trailing prose by extracting the first {...} block.
    const cleaned = raw
      .replace(/^```(?:json)?\s*/i, "")
      .replace(/\s*```$/i, "")
      .trim();
    const match = cleaned.match(/\{[\s\S]*\}/);
    const jsonStr = match ? match[0] : cleaned;

    const parsed = JSON.parse(jsonStr);

    if (parsed.confidence !== "high" && parsed.confidence !== "low") {
      return { confidence: "high", reason: "Grader returned invalid confidence." };
    }

    return {
      confidence: parsed.confidence,
      reason: typeof parsed.reason === "string" ? parsed.reason.slice(0, 200) : "",
    };
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : String(err);
    console.log(`   ⚠️  Grader failed: ${msg.slice(0, 400)}`);
    // Graceful degrade: treat as high-confidence so we don't block the user
    return { confidence: "high", reason: "Grader unavailable." };
  }
}

// ── Query Rewriter ────────────────────────────────────────────

/**
 * Rewrites the user's question to be more specific, use different
 * keywords, or target the core concept — so a second retrieval
 * pass has a better chance of finding relevant chunks.
 *
 * Returns the rewritten query. If rewriting fails or produces a
 * useless result, returns the original question unchanged.
 */
export async function rewriteQuery(
  originalQuery: string,
  weakReason: string
): Promise<string> {
  const prompt = `You rewrite search queries for a PDF Q&A system. The user's original question retrieved weak results.

Original question: "${originalQuery}"
Retrieval issue: ${weakReason}

Rewrite the question to improve retrieval. Strategies:
- Use different / more specific keywords.
- Focus on the core concept, dropping filler words.
- Expand abbreviations or use synonyms.
- If the original is vague, make it concrete.

Output ONLY the rewritten question — no explanation, no quotes, no prefix. Under 20 words.`;

  try {
    const result = await rewriterModel.generateContent(prompt);
    let rewritten = result.response.text().trim();

    // Strip quotes or common prefixes the model might add
    rewritten = rewritten.replace(/^["'`]|["'`]$/g, "").trim();
    rewritten = rewritten.replace(/^(rewritten question:|query:|question:)\s*/i, "").trim();

    if (!rewritten || rewritten.length < 3 || rewritten.length > 300) {
      return originalQuery;
    }

    // If the model just echoed the original, no point retrying
    if (rewritten.toLowerCase() === originalQuery.toLowerCase()) {
      return originalQuery;
    }

    return rewritten;
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : String(err);
    console.log(`   ⚠️  Rewriter failed (${msg.slice(0, 80)}), using original query`);
    return originalQuery;
  }
}
