/**
 * vectorStore.ts — MongoDB Atlas Vector & Text Search
 *
 * Replaces the in-memory JS array with MongoDB-backed storage.
 * Uses Atlas Vector Search ($vectorSearch) for semantic search
 * and Atlas Search ($search) for full-text keyword search.
 *
 * Falls back to brute-force search if Atlas indexes don't exist.
 */

import mongoose from "mongoose";
import Chunk, { IChunk } from "../models/Chunk";
import { ChunkMetadata } from "../services/pdfService";

// ── Types ──────────────────────────────────────────────────────

export interface VectorEntry {
  text: string;
  embedding: number[];
  metadata: ChunkMetadata;
}

export interface SearchResult {
  text: string;
  score: number;
  metadata: ChunkMetadata;
  matchType: "semantic" | "keyword" | "hybrid";
}

// ── CRUD Operations ────────────────────────────────────────────

export async function addChunks(documentId: string, entries: VectorEntry[]): Promise<void> {
  const docs = entries.map((e) => ({
    documentId: new mongoose.Types.ObjectId(documentId),
    text: e.text,
    embedding: e.embedding,
    pageNumber: e.metadata.pageNumber,
    sectionHeading: e.metadata.sectionHeading,
    chunkIndex: e.metadata.chunkIndex,
  }));

  await Chunk.insertMany(docs, { ordered: false });
}

export async function deleteChunksByDocument(documentId: string): Promise<void> {
  await Chunk.deleteMany({ documentId: new mongoose.Types.ObjectId(documentId) });
}

export async function getChunkCount(documentId?: string): Promise<number> {
  if (documentId) {
    return Chunk.countDocuments({ documentId: new mongoose.Types.ObjectId(documentId) });
  }
  return Chunk.countDocuments({});
}

// ── Atlas Vector Search ────────────────────────────────────────

async function vectorSearchAtlas(
  queryEmbedding: number[],
  documentId: string | null,
  topN: number
): Promise<SearchResult[]> {
  const filter = documentId
    ? { documentId: new mongoose.Types.ObjectId(documentId) }
    : {};

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const pipeline: any[] = [
    {
      $vectorSearch: {
        index: "chunk_vector_index",
        path: "embedding",
        queryVector: queryEmbedding,
        numCandidates: topN * 10,
        limit: topN,
        ...(documentId ? { filter } : {}),
      },
    },
    { $addFields: { score: { $meta: "vectorSearchScore" } } },
    {
      $project: {
        text: 1, pageNumber: 1, sectionHeading: 1, chunkIndex: 1, score: 1,
      },
    },
  ];

  const results = await Chunk.aggregate(pipeline);

  return results.map((r: Record<string, unknown>) => ({
    text: r.text as string,
    score: r.score as number,
    metadata: {
      pageNumber: r.pageNumber as number,
      sectionHeading: (r.sectionHeading as string) || null,
      chunkIndex: r.chunkIndex as number,
    },
    matchType: "semantic" as const,
  }));
}

// ── Atlas Full-Text Search ─────────────────────────────────────

async function fullTextSearchAtlas(
  queryText: string,
  documentId: string | null,
  topN: number
): Promise<SearchResult[]> {
  const searchStage: Record<string, unknown> = documentId
    ? {
        $search: {
          index: "chunk_text_index",
          compound: {
            must: [{ text: { query: queryText, path: "text", fuzzy: { maxEdits: 1 } } }],
            filter: [{ equals: { path: "documentId", value: new mongoose.Types.ObjectId(documentId) } }],
          },
        },
      }
    : {
        $search: {
          index: "chunk_text_index",
          text: { query: queryText, path: "text", fuzzy: { maxEdits: 1 } },
        },
      };

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const pipeline: any[] = [
    searchStage,
    { $addFields: { score: { $meta: "searchScore" } } },
    { $limit: topN },
    {
      $project: {
        text: 1, pageNumber: 1, sectionHeading: 1, chunkIndex: 1, score: 1,
      },
    },
  ];

  const results = await Chunk.aggregate(pipeline);

  return results.map((r: Record<string, unknown>) => ({
    text: r.text as string,
    score: r.score as number,
    metadata: {
      pageNumber: r.pageNumber as number,
      sectionHeading: (r.sectionHeading as string) || null,
      chunkIndex: r.chunkIndex as number,
    },
    matchType: "keyword" as const,
  }));
}

// ── Brute-Force Fallbacks ──────────────────────────────────────

function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0, ma = 0, mb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i]; ma += a[i] * a[i]; mb += b[i] * b[i];
  }
  ma = Math.sqrt(ma); mb = Math.sqrt(mb);
  if (ma === 0 || mb === 0) return 0;
  return dot / (ma * mb);
}

const STOPWORDS = new Set([
  "a","an","the","is","are","was","were","be","been","being","have","has",
  "had","do","does","did","will","would","could","should","may","might",
  "shall","can","need","to","of","in","for","on","with","at","by","from",
  "as","into","through","during","before","after","above","below","between",
  "out","off","over","under","again","then","once","here","there","when",
  "where","why","how","all","each","every","both","few","more","most",
  "other","some","such","no","nor","not","only","own","same","so","than",
  "too","very","just","because","but","and","or","if","while","about",
  "what","which","who","whom","this","that","these","those","i","me","my",
  "we","our","you","your","he","him","his","she","her","it","its","they",
  "them","their",
]);

export function tokenize(text: string): string[] {
  return text.toLowerCase().replace(/[^\w\s]/g, " ").split(/\s+/)
    .filter((w) => w.length > 1 && !STOPWORDS.has(w));
}

function bruteForceKeywordScore(query: string, chunkText: string): number {
  const qt = tokenize(query);
  if (qt.length === 0) return 0;
  const ct = new Set(tokenize(chunkText));
  if (ct.size === 0) return 0;
  const uniq = [...new Set(qt)];
  let matched = 0;
  for (const t of uniq) { if (ct.has(t)) matched++; }
  const coverage = matched / uniq.length;
  const exact = chunkText.toLowerCase().includes(query.toLowerCase().trim()) ? 0.3 : 0;
  return Math.min(coverage * 0.5 + exact, 1.0);
}

async function vectorSearchFallback(
  queryEmbedding: number[],
  documentId: string | null,
  topN: number
): Promise<SearchResult[]> {
  const filter = documentId ? { documentId: new mongoose.Types.ObjectId(documentId) } : {};
  const chunks = await Chunk.find(filter).lean<IChunk[]>();

  const scored = chunks.map((c) => ({
    text: c.text,
    score: cosineSimilarity(queryEmbedding, c.embedding),
    metadata: { pageNumber: c.pageNumber, sectionHeading: c.sectionHeading, chunkIndex: c.chunkIndex },
    matchType: "semantic" as const,
  }));

  return scored.sort((a, b) => b.score - a.score).slice(0, topN);
}

async function textSearchFallback(
  queryText: string,
  documentId: string | null,
  topN: number
): Promise<SearchResult[]> {
  const filter = documentId ? { documentId: new mongoose.Types.ObjectId(documentId) } : {};
  const chunks = await Chunk.find(filter).lean<IChunk[]>();

  const scored = chunks
    .map((c) => ({
      text: c.text,
      score: bruteForceKeywordScore(queryText, c.text),
      metadata: { pageNumber: c.pageNumber, sectionHeading: c.sectionHeading, chunkIndex: c.chunkIndex },
      matchType: "keyword" as const,
    }))
    .filter((s) => s.score > 0);

  return scored.sort((a, b) => b.score - a.score).slice(0, topN);
}

// ── Hybrid Search ──────────────────────────────────────────────

/**
 * Runs vector + text search (Atlas or fallback), merges with RRF.
 */
export async function hybridSearchMongo(
  queryEmbedding: number[],
  queryText: string,
  documentId: string | null,
  topN: number = 10
): Promise<SearchResult[]> {
  // Run both searches, falling back gracefully
  let vectorResults: SearchResult[];
  try {
    vectorResults = await vectorSearchAtlas(queryEmbedding, documentId, topN * 2);
    console.log(`   Atlas Vector Search: ${vectorResults.length} results`);
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : "";
    console.log(`   ⚠️  Atlas Vector Search unavailable (${msg.slice(0, 80)}), using fallback`);
    vectorResults = await vectorSearchFallback(queryEmbedding, documentId, topN * 2);
  }

  let textResults: SearchResult[];
  try {
    textResults = await fullTextSearchAtlas(queryText, documentId, topN * 2);
    console.log(`   Atlas Text Search: ${textResults.length} results`);
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : "";
    console.log(`   ⚠️  Atlas Text Search unavailable (${msg.slice(0, 80)}), using fallback`);
    textResults = await textSearchFallback(queryText, documentId, topN * 2);
  }

  // Reciprocal Rank Fusion
  const k = 60;
  const rrfMap = new Map<number, { score: number; result: SearchResult; vScore: number; tScore: number }>();

  vectorResults.forEach((r, rank) => {
    const key = r.metadata.chunkIndex;
    const contrib = 1 / (k + rank + 1);
    const existing = rrfMap.get(key);
    if (existing) { existing.score += contrib; existing.vScore = r.score; }
    else { rrfMap.set(key, { score: contrib, result: r, vScore: r.score, tScore: 0 }); }
  });

  textResults.forEach((r, rank) => {
    const key = r.metadata.chunkIndex;
    const contrib = 1 / (k + rank + 1);
    const existing = rrfMap.get(key);
    if (existing) { existing.score += contrib; existing.tScore = r.score; }
    else { rrfMap.set(key, { score: contrib, result: r, vScore: 0, tScore: r.score }); }
  });

  const merged = [...rrfMap.values()]
    .sort((a, b) => b.score - a.score)
    .slice(0, topN);

  return merged.map((item) => {
    let matchType: SearchResult["matchType"] = "hybrid";
    if (item.vScore > 0 && item.tScore === 0) matchType = "semantic";
    if (item.vScore === 0 && item.tScore > 0) matchType = "keyword";
    return { ...item.result, score: item.score, matchType };
  });
}
