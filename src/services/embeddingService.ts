/**
 * embeddingService.ts — Google Gemini Embeddings & Re-ranking
 *
 * Handles embedding generation and re-ranking only.
 * All search logic (vector, keyword, hybrid) has moved to
 * vectorStore.ts which uses MongoDB Atlas Search.
 */

import { GoogleGenerativeAI } from "@google/generative-ai";
import { SearchResult } from "../store/vectorStore";
import { tokenize } from "../store/vectorStore";

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY!);
const embeddingModel = genAI.getGenerativeModel({ model: "gemini-embedding-001" });

// ── Embedding Generation ───────────────────────────────────────

export async function getEmbedding(text: string): Promise<number[]> {
  const result = await embeddingModel.embedContent(text);
  return result.embedding.values;
}

export async function getEmbeddings(texts: string[]): Promise<number[][]> {
  const embeddings: number[][] = [];
  for (const text of texts) {
    const result = await embeddingModel.embedContent(text);
    embeddings.push(result.embedding.values);
  }
  return embeddings;
}

// ── Re-ranking ─────────────────────────────────────────────────

/**
 * Re-ranks hybrid search results using multiple signals:
 *   1. Original hybrid score (0.35)
 *   2. Query term coverage (0.25)
 *   3. Term proximity (0.20)
 *   4. Exact phrase match (0.20)
 */
export function rerankResults(
  queryText: string,
  candidates: SearchResult[],
  topN: number = 3
): SearchResult[] {
  if (candidates.length === 0) return [];

  const queryTokens = tokenize(queryText);
  const uniqueQueryTokens = [...new Set(queryTokens)];

  const reranked = candidates.map((candidate) => {
    const chunkTokens = tokenize(candidate.text);
    const chunkTokenSet = new Set(chunkTokens);

    const maxScore = candidates[0].score || 1;
    const normalizedHybrid = candidate.score / maxScore;

    let matchedTerms = 0;
    for (const qt of uniqueQueryTokens) {
      if (chunkTokenSet.has(qt)) matchedTerms++;
    }
    const coverage = uniqueQueryTokens.length > 0 ? matchedTerms / uniqueQueryTokens.length : 0;

    const proximity = computeProximity(queryTokens, chunkTokens);

    const exactBonus = candidate.text.toLowerCase().includes(queryText.toLowerCase().trim()) ? 1.0 : 0;

    const finalScore =
      normalizedHybrid * 0.35 +
      coverage * 0.25 +
      proximity * 0.20 +
      exactBonus * 0.20;

    return { ...candidate, score: finalScore };
  });

  return reranked.sort((a, b) => b.score - a.score).slice(0, topN);
}

function computeProximity(queryTokens: string[], chunkTokens: string[]): number {
  const uniqueQuery = [...new Set(queryTokens)];
  if (uniqueQuery.length === 0 || chunkTokens.length === 0) return 0;

  const positions: number[] = [];
  for (const qt of uniqueQuery) {
    for (let i = 0; i < chunkTokens.length; i++) {
      if (chunkTokens[i] === qt) { positions.push(i); break; }
    }
  }
  if (positions.length < 2) return positions.length > 0 ? 0.3 : 0;

  const sorted = positions.sort((a, b) => a - b);
  const windowSize = sorted[sorted.length - 1] - sorted[0] + 1;
  return Math.max(0, 1.0 - windowSize / chunkTokens.length);
}
