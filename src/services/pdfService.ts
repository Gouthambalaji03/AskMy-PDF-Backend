/**
 * pdfService.ts — PDF Text Extraction & Chunking
 *
 * =============================================================
 * RAG CONCEPT: Why do we split PDFs into chunks?
 * =============================================================
 *
 * Imagine you have a 200-page PDF textbook. If a user asks
 * "What is photosynthesis?", you don't need to send all 200
 * pages to the AI — that would be:
 *
 *   1. Too expensive (LLMs charge per token)
 *   2. Too slow (more tokens = slower response)
 *   3. Impossible (LLMs have a "context window" — a max number
 *      of tokens they can process at once. GPT-4o-mini can
 *      handle ~128K tokens, but sending a whole book wastes
 *      most of that space on irrelevant content)
 *
 * Instead, we split the PDF into small "chunks" (paragraphs
 * of ~500 characters). Later, when the user asks a question,
 * we find only the 3 most relevant chunks and send just those
 * to the AI. This is the "Retrieval" part of RAG.
 *
 * =============================================================
 * RAG CONCEPT: What is "overlap" in chunking?
 * =============================================================
 *
 * When we split text into chunks, a sentence might get cut in
 * half right at the boundary. By adding a 50-character overlap,
 * each chunk includes the last 50 characters of the previous
 * chunk. This way, sentences at the edges aren't lost and the
 * AI has better context for each chunk.
 *
 * Example without overlap:
 *   Chunk 1: "...the process of photosynthe"
 *   Chunk 2: "sis converts sunlight into..."
 *
 * Example WITH overlap:
 *   Chunk 1: "...the process of photosynthe"
 *   Chunk 2: "of photosynthesis converts sunlight into..."
 *   Now chunk 2 has the full word "photosynthesis"!
 * =============================================================
 */

import fs from "fs";
import pdfParse from "pdf-parse";

/**
 * Reads a PDF file from disk and extracts all its text content.
 * Returns the raw text as a single string.
 */
export async function extractTextFromPDF(filePath: string): Promise<string> {
  // Read the PDF file as a binary buffer
  const fileBuffer = fs.readFileSync(filePath);

  // pdf-parse reads the buffer and returns an object with the text
  const pdfData = await pdfParse(fileBuffer);

  // pdfData.text contains all the text from every page, concatenated
  return pdfData.text;
}

/**
 * Splits a long string of text into smaller chunks.
 *
 * @param text - The full text extracted from the PDF
 * @param chunkSize - How many characters per chunk (default: 500)
 * @param overlap - How many characters to repeat between chunks (default: 50)
 * @returns An array of text chunks
 *
 * Think of it like cutting a long ribbon into pieces, but each
 * piece slightly overlaps with the next one.
 */
export function splitTextIntoChunks(
  text: string,
  chunkSize: number = 500,
  overlap: number = 50
): string[] {
  const chunks: string[] = [];

  // Start at position 0 and move forward by (chunkSize - overlap) each time
  // The overlap means we step forward less than a full chunk, so chunks share edges
  let startIndex = 0;

  while (startIndex < text.length) {
    // Grab a slice of text from startIndex to startIndex + chunkSize
    const chunk = text.slice(startIndex, startIndex + chunkSize);

    // Only add non-empty chunks (skip if we're past the end)
    const trimmed = chunk.trim();
    if (trimmed.length > 0) {
      chunks.push(trimmed);
    }

    // Move the starting position forward — but not by the full chunk size,
    // so the next chunk will re-include the last `overlap` characters
    startIndex += chunkSize - overlap;
  }

  return chunks;
}
