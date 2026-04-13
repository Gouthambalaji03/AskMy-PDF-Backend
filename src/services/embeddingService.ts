/**
 * embeddingService.ts — Google Gemini Embeddings & Cosine Similarity
 *
 * =============================================================
 * RAG CONCEPT: What is an "embedding"?
 * =============================================================
 *
 * An embedding is a way to convert text into a list of numbers
 * (called a "vector") that captures the MEANING of that text.
 *
 * Think of it like GPS coordinates for meaning:
 *   - "I love dogs" → [0.2, 0.8, -0.1, ...]  (768 numbers)
 *   - "I adore puppies" → [0.21, 0.79, -0.09, ...]  (very similar!)
 *   - "The stock market crashed" → [-0.5, 0.1, 0.7, ...]  (very different)
 *
 * Even though "dogs" and "puppies" are different words, their
 * embeddings are very close together because they mean similar
 * things. This is what makes RAG work — we can find chunks that
 * are semantically related to the user's question, not just
 * chunks that contain the exact same keywords.
 *
 * We use Google's "text-embedding-004" model, which
 * converts any text into a vector of 768 numbers.
 *
 * =============================================================
 * RAG CONCEPT: What is "cosine similarity"?
 * =============================================================
 *
 * Once we have two embedding vectors, we need to measure how
 * "similar" they are. Cosine similarity does this by measuring
 * the angle between two vectors:
 *
 *   - If two vectors point in the SAME direction → similarity = 1.0
 *     (the texts mean very similar things)
 *   - If two vectors are perpendicular → similarity = 0.0
 *     (the texts are unrelated)
 *   - If two vectors point in OPPOSITE directions → similarity = -1.0
 *     (the texts mean opposite things)
 *
 * The formula is:
 *   cosine_similarity = (A · B) / (|A| × |B|)
 *
 * Where:
 *   A · B = sum of (a[i] * b[i]) for each dimension (dot product)
 *   |A|   = square root of sum of (a[i]²) (magnitude/length)
 *
 * Why cosine and not just distance? Because cosine similarity
 * cares about the DIRECTION of the vectors, not their length.
 * This makes it more robust for comparing text embeddings.
 * =============================================================
 */

import { GoogleGenerativeAI } from "@google/generative-ai";

// Initialize the Gemini client with the API key from .env
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY!);

// Get the embedding model instance
const embeddingModel = genAI.getGenerativeModel({ model: "gemini-embedding-001" });

/**
 * Takes a piece of text and returns its embedding vector
 * (an array of 768 numbers that represent its meaning).
 *
 * @param text - Any text string (a chunk, a user question, etc.)
 * @returns A number array of length 768
 */
export async function getEmbedding(text: string): Promise<number[]> {
  const result = await embeddingModel.embedContent(text);
  return result.embedding.values;
}

/**
 * Takes an array of text strings and returns all their embeddings.
 * Embeds each chunk individually using embedContent.
 *
 * @param texts - Array of text chunks to embed
 * @returns Array of embedding vectors (same order as input)
 */
export async function getEmbeddings(texts: string[]): Promise<number[][]> {
  const embeddings: number[][] = [];

  for (const text of texts) {
    const result = await embeddingModel.embedContent(text);
    embeddings.push(result.embedding.values);
  }

  return embeddings;
}

/**
 * Computes cosine similarity between two vectors.
 *
 * Returns a number between -1 and 1:
 *   1.0  = identical meaning
 *   0.0  = completely unrelated
 *  -1.0  = opposite meaning
 *
 * @param vecA - First embedding vector
 * @param vecB - Second embedding vector
 * @returns Similarity score
 */
export function cosineSimilarity(vecA: number[], vecB: number[]): number {
  // Step 1: Compute the dot product (multiply corresponding elements and sum)
  let dotProduct = 0;
  let magnitudeA = 0;
  let magnitudeB = 0;

  for (let i = 0; i < vecA.length; i++) {
    dotProduct += vecA[i] * vecB[i]; // A · B
    magnitudeA += vecA[i] * vecA[i]; // |A|² (sum of squares)
    magnitudeB += vecB[i] * vecB[i]; // |B|² (sum of squares)
  }

  // Step 2: Take square roots to get actual magnitudes
  magnitudeA = Math.sqrt(magnitudeA);
  magnitudeB = Math.sqrt(magnitudeB);

  // Avoid division by zero (shouldn't happen with real embeddings)
  if (magnitudeA === 0 || magnitudeB === 0) return 0;

  // Step 3: Divide dot product by the product of magnitudes
  return dotProduct / (magnitudeA * magnitudeB);
}

/**
 * Given a query embedding and a list of stored chunks with embeddings,
 * find the top N most similar chunks.
 *
 * This is the core "retrieval" step of RAG:
 *   User question → embedding → compare against all chunks → return best matches
 *
 * @param queryEmbedding - The embedding of the user's question
 * @param chunks - All stored chunks with their embeddings
 * @param topN - How many results to return (default: 3)
 * @returns The top N most relevant chunks, sorted by similarity
 */
export function findTopMatches(
  queryEmbedding: number[],
  chunks: { text: string; embedding: number[] }[],
  topN: number = 3
): { text: string; score: number }[] {
  // Score each chunk by its similarity to the query
  const scored = chunks.map((chunk) => ({
    text: chunk.text,
    score: cosineSimilarity(queryEmbedding, chunk.embedding),
  }));

  // Sort by score (highest first) and take the top N
  return scored.sort((a, b) => b.score - a.score).slice(0, topN);
}
