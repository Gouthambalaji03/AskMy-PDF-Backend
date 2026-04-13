/**
 * pdfService.ts — PDF Text Extraction & Smart Recursive Chunking
 *
 * =============================================================
 * TIER 2: Smarter Chunking + Chunk Metadata
 * =============================================================
 *
 * The old approach used fixed-size character splitting (every 500
 * characters, with 50-char overlap). This is simple but dumb —
 * it cuts sentences and paragraphs in half, losing context.
 *
 * The new approach uses RECURSIVE TEXT SPLITTING:
 *   1. First, try to split by PARAGRAPHS (double newline)
 *   2. If a paragraph is still too large, split by SENTENCES
 *   3. If a sentence is still too large, fall back to fixed-size
 *
 * This keeps semantic units (paragraphs, sentences) intact as
 * much as possible, which means the embedding for each chunk
 * more accurately represents its meaning.
 *
 * We also track METADATA for each chunk:
 *   - pageNumber: which page of the PDF this chunk came from
 *   - sectionHeading: the nearest section heading (if detected)
 *   - chunkIndex: the chunk's position in the document
 *
 * This metadata lets the LLM cite specific pages and sections
 * in its answers, making responses more trustworthy.
 * =============================================================
 */

import fs from "fs";
import pdfParse from "pdf-parse";

// ── Types ──────────────────────────────────────────────────────

export interface ChunkMetadata {
  pageNumber: number;
  sectionHeading: string | null;
  chunkIndex: number;
}

export interface ChunkWithMetadata {
  text: string;
  metadata: ChunkMetadata;
}

// ── PDF Text Extraction ────────────────────────────────────────

/**
 * Reads a PDF and extracts its text along with page count.
 * The page count is used to estimate which page each chunk
 * belongs to based on character position.
 */
export async function extractTextFromPDF(
  filePath: string
): Promise<{ text: string; numPages: number }> {
  const fileBuffer = fs.readFileSync(filePath);
  const pdfData = await pdfParse(fileBuffer);
  return { text: pdfData.text, numPages: pdfData.numpages };
}

// ── Section Heading Detection ──────────────────────────────────

/**
 * Scans a block of text and tries to detect section headings.
 * Looks for common patterns like:
 *   - "Chapter 1: Introduction"
 *   - "1.2 Methods and Materials"
 *   - "INTRODUCTION" (all-caps short lines)
 *   - "Section 3 — Results"
 *
 * Returns the first heading found, or null if none detected.
 */
function detectSectionHeading(text: string): string | null {
  const lines = text.split("\n");
  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.length < 3) continue;

    // "Chapter 1", "Section 2.1", "Part III"
    if (/^(chapter|section|part)\s+[\dIVXivx]/i.test(trimmed) && trimmed.length < 120) {
      return trimmed;
    }
    // "1.2 Title Text" or "1.2.3 Sub-section"
    if (/^\d+(\.\d+)+\s+[A-Z]/.test(trimmed) && trimmed.length < 120) {
      return trimmed;
    }
    // "1 Introduction" (single number + title)
    if (/^\d+\s+[A-Z][a-zA-Z\s]+$/.test(trimmed) && trimmed.length < 80) {
      return trimmed;
    }
    // "INTRODUCTION" — all-caps lines that look like headings
    if (/^[A-Z][A-Z\s&:-]{3,}$/.test(trimmed) && trimmed.length < 80) {
      return trimmed;
    }
  }
  return null;
}

/**
 * Estimates the page number for a chunk based on its character
 * position within the full document text.
 */
function estimatePageNumber(
  charPosition: number,
  totalChars: number,
  numPages: number
): number {
  if (numPages <= 1) return 1;
  if (totalChars === 0) return 1;
  const charsPerPage = totalChars / numPages;
  return Math.min(Math.ceil((charPosition + 1) / charsPerPage), numPages);
}

// ── Recursive Text Splitting ───────────────────────────────────

/**
 * The core recursive splitting algorithm.
 *
 * Strategy:
 *   1. If the text fits within maxSize, return it as-is
 *   2. Try splitting by paragraphs (double newline)
 *   3. If paragraphs are still too big, try splitting by sentences
 *   4. Last resort: fixed-size splitting with overlap
 *
 * At each level, small adjacent pieces are merged together to
 * avoid creating tiny chunks (which would have poor embeddings).
 */
function recursiveSplit(
  text: string,
  maxSize: number,
  overlap: number
): string[] {
  const trimmed = text.trim();
  if (trimmed.length === 0) return [];
  if (trimmed.length <= maxSize) return [trimmed];

  // Level 1: Split by paragraphs (double newline)
  const paragraphs = trimmed.split(/\n\s*\n/).filter((p) => p.trim().length > 0);
  if (paragraphs.length > 1) {
    return mergeAndSplit(paragraphs, maxSize, overlap);
  }

  // Level 2: Split by sentences
  const sentences = trimmed.match(/[^.!?]*[.!?]+(?:\s|$)|[^.!?]+$/g);
  if (sentences && sentences.length > 1) {
    return mergeAndSplit(
      sentences.filter((s) => s.trim().length > 0),
      maxSize,
      overlap
    );
  }

  // Level 3: Fixed-size fallback
  return fixedSizeChunk(trimmed, maxSize, overlap);
}

/**
 * Merges small text pieces into chunks up to maxSize, then
 * recursively splits any pieces that are still too large.
 *
 * Adds overlap between adjacent chunks so context isn't lost
 * at the boundaries.
 */
function mergeAndSplit(
  pieces: string[],
  maxSize: number,
  overlap: number
): string[] {
  const chunks: string[] = [];
  let current = "";

  for (const piece of pieces) {
    const trimmedPiece = piece.trim();
    if (!trimmedPiece) continue;

    // Can we fit this piece into the current chunk?
    const separator = current ? "\n\n" : "";
    if (current.length + separator.length + trimmedPiece.length <= maxSize) {
      current = current + separator + trimmedPiece;
    } else {
      // Save current chunk if it has content
      if (current.trim()) {
        chunks.push(current.trim());
      }

      // If this single piece is too large, recurse deeper
      if (trimmedPiece.length > maxSize) {
        const subChunks = recursiveSplit(trimmedPiece, maxSize, overlap);
        chunks.push(...subChunks);
        current = "";
      } else {
        // Start new chunk with overlap from previous
        if (chunks.length > 0 && overlap > 0) {
          const prevChunk = chunks[chunks.length - 1];
          const overlapText = prevChunk.slice(-overlap).trim();
          const candidate = overlapText + " " + trimmedPiece;
          current = candidate.length <= maxSize ? candidate : trimmedPiece;
        } else {
          current = trimmedPiece;
        }
      }
    }
  }

  if (current.trim()) {
    chunks.push(current.trim());
  }

  return chunks;
}

/**
 * Last-resort fixed-size chunking with overlap.
 * Only used when text can't be split by paragraphs or sentences.
 */
function fixedSizeChunk(
  text: string,
  chunkSize: number,
  overlap: number
): string[] {
  const chunks: string[] = [];
  let start = 0;

  while (start < text.length) {
    const chunk = text.slice(start, start + chunkSize).trim();
    if (chunk.length > 0) chunks.push(chunk);
    start += chunkSize - overlap;
  }

  return chunks;
}

// ── Main Processing Function ───────────────────────────────────

/**
 * The full document processing pipeline:
 *   1. Extract text from PDF (with page count)
 *   2. Recursively split into smart chunks
 *   3. Attach metadata (page number, section heading, index)
 *
 * @param filePath - Path to the uploaded PDF file
 * @param chunkSize - Target maximum characters per chunk (default: 500)
 * @param overlap - Characters of overlap between chunks (default: 50)
 * @returns Array of chunks with metadata
 */
export async function processDocument(
  filePath: string,
  chunkSize: number = 500,
  overlap: number = 50
): Promise<{ chunks: ChunkWithMetadata[]; totalChars: number }> {
  const { text, numPages } = await extractTextFromPDF(filePath);

  if (!text || text.trim().length === 0) {
    return { chunks: [], totalChars: 0 };
  }

  const totalChars = text.length;
  const rawChunks = recursiveSplit(text, chunkSize, overlap);

  // Track character position to estimate page numbers.
  // We search for each chunk in the original text starting from
  // where the previous chunk ended.
  let searchFrom = 0;
  let lastHeading: string | null = null;

  const chunks: ChunkWithMetadata[] = rawChunks.map((chunkText, index) => {
    // Find this chunk's approximate position in the original text
    let charPos = text.indexOf(chunkText.slice(0, 80), searchFrom);
    if (charPos === -1) charPos = searchFrom; // fallback
    searchFrom = charPos + 1;

    // Detect section heading within this chunk
    const heading = detectSectionHeading(chunkText);
    if (heading) lastHeading = heading;

    return {
      text: chunkText,
      metadata: {
        pageNumber: estimatePageNumber(charPos, totalChars, numPages),
        sectionHeading: heading || lastHeading,
        chunkIndex: index,
      },
    };
  });

  return { chunks, totalChars };
}
