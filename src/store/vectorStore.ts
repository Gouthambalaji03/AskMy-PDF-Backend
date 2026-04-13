/**
 * vectorStore.ts — In-Memory Vector Storage
 *
 * =============================================================
 * RAG CONCEPT: What is a "vector store"?
 * =============================================================
 *
 * After we split the PDF into chunks and generate an embedding
 * for each chunk, we need to STORE those embeddings somewhere
 * so we can search through them later.
 *
 * In production, you'd use a dedicated vector database like
 * Pinecone, Weaviate, or Qdrant. But for learning purposes,
 * a simple JavaScript array works perfectly!
 *
 * Each entry in our store looks like:
 *   { text: "the actual chunk text", embedding: [0.1, -0.3, ...] }
 *
 * When a user asks a question, we:
 *   1. Convert their question to an embedding
 *   2. Compare it against every stored embedding
 *   3. Return the chunks whose embeddings are most similar
 * =============================================================
 */

/**
 * Each chunk is stored as its original text paired with its
 * embedding vector (an array of numbers).
 */
export interface VectorEntry {
  text: string;
  embedding: number[];
}

/**
 * Our "database" — just an array in memory.
 *
 * Pros: Simple, no setup, great for learning
 * Cons: Data is lost when the server restarts,
 *       doesn't scale to millions of chunks
 *
 * We'll show how to swap this with Pinecone later!
 */
let vectorStore: VectorEntry[] = [];

/** Add multiple chunk+embedding pairs to the store */
export function addToStore(entries: VectorEntry[]): void {
  vectorStore.push(...entries);
}

/** Get all entries in the store (used for similarity search) */
export function getStore(): VectorEntry[] {
  return vectorStore;
}

/** Clear the store (useful when uploading a new PDF) */
export function clearStore(): void {
  vectorStore = [];
}

/** Check how many chunks are currently stored */
export function getStoreSize(): number {
  return vectorStore.length;
}
