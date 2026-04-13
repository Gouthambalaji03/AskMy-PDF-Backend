/**
 * embeddingService.ts — Embeddings, Hybrid Search & Re-ranking
 *
 * =============================================================
 * TIER 2: Hybrid Search + Re-ranking
 * =============================================================
 *
 * HYBRID SEARCH: Combines two search strategies:
 *
 *   1. VECTOR SEARCH (semantic): Uses embedding cosine similarity.
 *      Great at understanding meaning — "dogs" matches "puppies".
 *      But can miss exact terms like "Table 4.2" or "GDP".
 *
 *   2. KEYWORD SEARCH (lexical): Uses term matching and frequency.
 *      Great at finding exact terms, abbreviations, and numbers.
 *      But misses synonyms — "dogs" won't match "puppies".
 *
 *   By combining both, we get the best of both worlds. The scores
 *   are merged using Reciprocal Rank Fusion (RRF), which weights
 *   results by their rank position in each list.
 *
 * RE-RANKING: After hybrid search returns the top candidates,
 *   we re-score them with a more detailed analysis:
 *
 *   - Query term coverage (what % of query words appear?)
 *   - Proximity (are matching terms close together?)
 *   - Exact phrase match bonus
 *   - Original hybrid score
 *
 *   This two-stage approach (fast retrieval → precise re-ranking)
 *   is the same pattern used by production search engines.
 * =============================================================
 */

import { GoogleGenerativeAI } from "@google/generative-ai";
import { ChunkMetadata } from "./pdfService";
import { VectorEntry, searchByKeyword, tokenize } from "../store/vectorStore";

// ── Gemini Embedding Client ────────────────────────────────────

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY!);
const embeddingModel = genAI.getGenerativeModel({ model: "gemini-embedding-001" });

// ── Core Embedding Functions ───────────────────────────────────

/** Convert a single text string into a 768-dimension embedding vector */
export async function getEmbedding(text: string): Promise<number[]> {
  const result = await embeddingModel.embedContent(text);
  return result.embedding.values;
}

/** Convert an array of texts into embedding vectors (sequential) */
export async function getEmbeddings(texts: string[]): Promise<number[][]> {
  const embeddings: number[][] = [];
  for (const text of texts) {
    const result = await embeddingModel.embedContent(text);
    embeddings.push(result.embedding.values);
  }
  return embeddings;
}

// ── Cosine Similarity ──────────────────────────────────────────

/**
 * Computes cosine similarity between two vectors.
 * Returns a number between -1 (opposite) and 1 (identical).
 */
export function cosineSimilarity(vecA: number[], vecB: number[]): number {
  let dotProduct = 0;
  let magnitudeA = 0;
  let magnitudeB = 0;

  for (let i = 0; i < vecA.length; i++) {
    dotProduct += vecA[i] * vecB[i];
    magnitudeA += vecA[i] * vecA[i];
    magnitudeB += vecB[i] * vecB[i];
  }

  magnitudeA = Math.sqrt(magnitudeA);
  magnitudeB = Math.sqrt(magnitudeB);

  if (magnitudeA === 0 || magnitudeB === 0) return 0;
  return dotProduct / (magnitudeA * magnitudeB);
}

// ── Search Result Type ─────────────────────────────────────────

export interface SearchResult {
  text: string;
  score: number;
  metadata: ChunkMetadata;
  matchType: "semantic" | "keyword" | "hybrid";
}

// ── Vector Search ──────────────────────────────────────────────

/**
 * Pure vector (semantic) search — finds chunks whose embeddings
 * are closest to the query embedding using cosine similarity.
 */
function vectorSearch(
  queryEmbedding: number[],
  chunks: VectorEntry[],
  topN: number
): { entry: VectorEntry; score: number }[] {
  const scored = chunks.map((chunk) => ({
    entry: chunk,
    score: cosineSimilarity(queryEmbedding, chunk.embedding),
  }));

  return scored.sort((a, b) => b.score - a.score).slice(0, topN);
}

// ── Hybrid Search ──────────────────────────────────────────────

/**
 * Combines vector search and keyword search using Reciprocal
 * Rank Fusion (RRF).
 *
 * RRF merges two ranked lists by scoring each result as:
 *   RRF_score = 1/(k + rank_in_list)
 *
 * Where k is a constant (typically 60). This gives higher weight
 * to results that rank well in BOTH lists, without needing to
 * normalize the raw scores (which are on different scales).
 *
 * @param queryEmbedding - The question's embedding vector
 * @param queryText - The raw question text (for keyword search)
 * @param chunks - All stored chunks with embeddings
 * @param topN - How many results to return
 * @returns Merged and scored results
 */
export function hybridSearch(
  queryEmbedding: number[],
  queryText: string,
  chunks: VectorEntry[],
  topN: number = 10
): SearchResult[] {
  const k = 60; // RRF constant — standard value from literature

  // Get top results from both search methods
  const vectorResults = vectorSearch(queryEmbedding, chunks, topN * 2);
  const keywordResults = searchByKeyword(queryText, topN * 2);

  // Build RRF score map: chunkIndex → combined score
  const rrfScores = new Map<number, { score: number; entry: VectorEntry; vectorScore: number; keywordScore: number }>();

  // Add vector search contributions
  vectorResults.forEach((result, rank) => {
    const idx = result.entry.metadata.chunkIndex;
    const existing = rrfScores.get(idx);
    const rrfContribution = 1 / (k + rank + 1);
    if (existing) {
      existing.score += rrfContribution;
      existing.vectorScore = result.score;
    } else {
      rrfScores.set(idx, {
        score: rrfContribution,
        entry: result.entry,
        vectorScore: result.score,
        keywordScore: 0,
      });
    }
  });

  // Add keyword search contributions
  keywordResults.forEach((result, rank) => {
    const idx = result.entry.metadata.chunkIndex;
    const existing = rrfScores.get(idx);
    const rrfContribution = 1 / (k + rank + 1);
    if (existing) {
      existing.score += rrfContribution;
      existing.keywordScore = result.score;
    } else {
      rrfScores.set(idx, {
        score: rrfContribution,
        entry: result.entry,
        vectorScore: 0,
        keywordScore: result.score,
      });
    }
  });

  // Sort by combined RRF score and return top N
  const merged = [...rrfScores.values()]
    .sort((a, b) => b.score - a.score)
    .slice(0, topN);

  return merged.map((item) => {
    let matchType: SearchResult["matchType"] = "hybrid";
    if (item.vectorScore > 0 && item.keywordScore === 0) matchType = "semantic";
    if (item.vectorScore === 0 && item.keywordScore > 0) matchType = "keyword";

    return {
      text: item.entry.text,
      score: item.score,
      metadata: item.entry.metadata,
      matchType,
    };
  });
}

// ── Re-ranking ─────────────────────────────────────────────────

/**
 * Re-ranks hybrid search results using multiple fine-grained
 * signals. This is more expensive than initial retrieval, so
 * we only apply it to the top candidates.
 *
 * Scoring signals:
 *   1. Original hybrid score (0.35 weight)
 *   2. Query term coverage — what fraction of query terms appear (0.25 weight)
 *   3. Term proximity — how close together matching terms are (0.20 weight)
 *   4. Exact phrase match — bonus for containing the exact query (0.20 weight)
 *
 * @param queryText - The original question text
 * @param candidates - Results from hybrid search
 * @param topN - How many final results to return
 * @returns Re-ranked and scored results
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

    // Signal 1: Original hybrid score (normalize to 0-1 range)
    const maxScore = candidates[0].score || 1;
    const normalizedHybrid = candidate.score / maxScore;

    // Signal 2: Query term coverage
    let matchedTerms = 0;
    for (const qt of uniqueQueryTokens) {
      if (chunkTokenSet.has(qt)) matchedTerms++;
    }
    const coverage = uniqueQueryTokens.length > 0
      ? matchedTerms / uniqueQueryTokens.length
      : 0;

    // Signal 3: Term proximity — how close matching terms are to each other
    const proximity = computeProximity(queryTokens, chunkTokens);

    // Signal 4: Exact phrase match
    const queryLower = queryText.toLowerCase().trim();
    const chunkLower = candidate.text.toLowerCase();
    const exactBonus = chunkLower.includes(queryLower) ? 1.0 : 0;

    // Weighted combination
    const finalScore =
      normalizedHybrid * 0.35 +
      coverage * 0.25 +
      proximity * 0.20 +
      exactBonus * 0.20;

    return { ...candidate, score: finalScore };
  });

  return reranked
    .sort((a, b) => b.score - a.score)
    .slice(0, topN);
}

/**
 * Measures how close together the matching query terms appear
 * within the chunk text. A higher score means the terms are
 * clustered together (likely a more relevant passage).
 *
 * Algorithm:
 *   1. Find the positions of all matching query terms in the chunk
 *   2. Compute the minimum window that contains all matches
 *   3. Score inversely proportional to window size
 *
 * Returns 0.0 (terms far apart / not found) to 1.0 (tightly clustered).
 */
function computeProximity(queryTokens: string[], chunkTokens: string[]): number {
  const uniqueQuery = [...new Set(queryTokens)];
  if (uniqueQuery.length === 0 || chunkTokens.length === 0) return 0;

  // Find positions of each query term in the chunk
  const positions: number[] = [];
  for (const qt of uniqueQuery) {
    for (let i = 0; i < chunkTokens.length; i++) {
      if (chunkTokens[i] === qt) {
        positions.push(i);
        break; // Take first occurrence only
      }
    }
  }

  // If fewer than 2 terms found, can't measure proximity
  if (positions.length < 2) {
    return positions.length > 0 ? 0.3 : 0;
  }

  // Window size = distance between first and last matching term
  const sorted = positions.sort((a, b) => a - b);
  const windowSize = sorted[sorted.length - 1] - sorted[0] + 1;

  // Normalize: smaller window (relative to chunk size) = higher score
  const ratio = windowSize / chunkTokens.length;
  return Math.max(0, 1.0 - ratio);
}

// ── Legacy Export (backward compatibility) ─────────────────────

/** @deprecated Use hybridSearch + rerankResults instead */
export function findTopMatches(
  queryEmbedding: number[],
  chunks: { text: string; embedding: number[] }[],
  topN: number = 3
): { text: string; score: number }[] {
  const scored = chunks.map((chunk) => ({
    text: chunk.text,
    score: cosineSimilarity(queryEmbedding, chunk.embedding),
  }));
  return scored.sort((a, b) => b.score - a.score).slice(0, topN);
}
