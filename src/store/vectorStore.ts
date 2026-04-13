/**
 * vectorStore.ts — In-Memory Vector Storage with Keyword Search
 *
 * =============================================================
 * TIER 2: Metadata Storage + Keyword Search Support
 * =============================================================
 *
 * Updated to store chunk metadata (page number, section heading)
 * alongside text and embeddings.
 *
 * Also adds a keyword search function used by the hybrid search
 * system — it finds chunks that contain the user's query terms,
 * even when the semantic embedding search might miss exact
 * keywords like "Section 4.2" or "GDP".
 * =============================================================
 */

import { ChunkMetadata } from "../services/pdfService";

// ── Types ──────────────────────────────────────────────────────

export interface VectorEntry {
  text: string;
  embedding: number[];
  metadata: ChunkMetadata;
}

// ── Store ──────────────────────────────────────────────────────

let vectorStore: VectorEntry[] = [];

/** Add multiple chunk+embedding pairs to the store */
export function addToStore(entries: VectorEntry[]): void {
  vectorStore.push(...entries);
}

/** Get all entries in the store */
export function getStore(): VectorEntry[] {
  return vectorStore;
}

/** Clear the store (when uploading a new PDF) */
export function clearStore(): void {
  vectorStore = [];
}

/** Check how many chunks are currently stored */
export function getStoreSize(): number {
  return vectorStore.length;
}

// ── Keyword Search ─────────────────────────────────────────────

/** Stopwords to ignore during keyword matching */
const STOPWORDS = new Set([
  "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
  "have", "has", "had", "do", "does", "did", "will", "would", "could",
  "should", "may", "might", "shall", "can", "need", "dare", "ought",
  "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
  "as", "into", "through", "during", "before", "after", "above",
  "below", "between", "out", "off", "over", "under", "again", "further",
  "then", "once", "here", "there", "when", "where", "why", "how", "all",
  "each", "every", "both", "few", "more", "most", "other", "some",
  "such", "no", "nor", "not", "only", "own", "same", "so", "than",
  "too", "very", "just", "because", "but", "and", "or", "if", "while",
  "about", "what", "which", "who", "whom", "this", "that", "these",
  "those", "i", "me", "my", "we", "our", "you", "your", "he", "him",
  "his", "she", "her", "it", "its", "they", "them", "their",
]);

/**
 * Tokenizes text into meaningful words:
 *   - Lowercases everything
 *   - Removes punctuation
 *   - Filters out stopwords and single characters
 */
export function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^\w\s]/g, " ")
    .split(/\s+/)
    .filter((word) => word.length > 1 && !STOPWORDS.has(word));
}

/**
 * Computes keyword relevance score for a chunk against a query.
 *
 * Scoring factors:
 *   1. Term frequency: how many query terms appear in the chunk
 *   2. Coverage: what fraction of unique query terms are found
 *   3. Exact phrase bonus: extra score if the exact query appears
 *   4. Density: query term matches / total chunk words
 *
 * Returns a score between 0 and 1.
 */
export function keywordScore(query: string, chunkText: string): number {
  const queryTokens = tokenize(query);
  if (queryTokens.length === 0) return 0;

  const chunkTokens = tokenize(chunkText);
  if (chunkTokens.length === 0) return 0;

  const chunkTokenSet = new Set(chunkTokens);
  const uniqueQueryTokens = [...new Set(queryTokens)];

  // 1. Coverage: fraction of unique query terms found in chunk
  let matchedTerms = 0;
  for (const qt of uniqueQueryTokens) {
    if (chunkTokenSet.has(qt)) matchedTerms++;
  }
  const coverage = matchedTerms / uniqueQueryTokens.length;

  // 2. Term frequency: total matches / chunk length (density)
  let totalMatches = 0;
  for (const qt of queryTokens) {
    for (const ct of chunkTokens) {
      if (ct === qt) totalMatches++;
    }
  }
  const density = Math.min(totalMatches / chunkTokens.length, 1);

  // 3. Exact phrase bonus
  const queryLower = query.toLowerCase().trim();
  const chunkLower = chunkText.toLowerCase();
  const exactMatch = chunkLower.includes(queryLower) ? 0.3 : 0;

  // Weighted combination (coverage is most important)
  const score = coverage * 0.5 + density * 0.2 + exactMatch;

  return Math.min(score, 1.0);
}

/**
 * Searches all chunks by keyword relevance.
 * Returns entries sorted by keyword score (highest first).
 */
export function searchByKeyword(
  query: string,
  topN: number = 10
): { entry: VectorEntry; score: number }[] {
  const scored = vectorStore.map((entry) => ({
    entry,
    score: keywordScore(query, entry.text),
  }));

  return scored
    .filter((s) => s.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, topN);
}
