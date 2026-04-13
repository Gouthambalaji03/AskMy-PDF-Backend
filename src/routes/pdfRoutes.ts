/**
 * pdfRoutes.ts — API Routes for PDF Upload, Query, and Chat
 *
 * =============================================================
 * TIER 2: Full Upgraded RAG Pipeline
 * =============================================================
 *
 * Upload flow (/api/upload):
 *   1. Receive PDF → extract text with page count
 *   2. SMART CHUNKING: recursive split (paragraphs → sentences → fixed)
 *   3. Attach METADATA: page number, section heading, chunk index
 *   4. Generate embeddings for all chunks
 *   5. Store { text, embedding, metadata } in vector store
 *
 * Query flow (/api/ask):
 *   1. Embed the user's question
 *   2. HYBRID SEARCH: combine vector + keyword search via RRF
 *   3. RE-RANK: score top candidates with multiple signals
 *   4. Send top 3 chunks (with metadata) to Gemini
 *   5. Return answer with page numbers, sections, and match types
 * =============================================================
 */

import { Router, Request, Response } from "express";
import multer from "multer";
import path from "path";
import fs from "fs";
import { processDocument } from "../services/pdfService";
import { getEmbeddings, getEmbedding, hybridSearch, rerankResults } from "../services/embeddingService";
import { addToStore, clearStore, getStore, getStoreSize } from "../store/vectorStore";
import { askQuestion } from "../services/chatService";

const router = Router();

// ── Multer File Upload Config ──────────────────────────────────

const storage = multer.diskStorage({
  destination: (_req, _file, cb) => {
    const uploadDir = path.join(__dirname, "../../uploads");
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir, { recursive: true });
    }
    cb(null, uploadDir);
  },
  filename: (_req, file, cb) => {
    const uniqueName = `${Date.now()}-${file.originalname}`;
    cb(null, uniqueName);
  },
});

const upload = multer({
  storage,
  limits: {
    fileSize: parseInt(process.env.MAX_FILE_SIZE || "10485760"), // 10MB default
  },
  fileFilter: (_req, file, cb) => {
    if (file.mimetype === "application/pdf") {
      cb(null, true);
    } else {
      cb(new Error("Only PDF files are allowed"));
    }
  },
});

// ── POST /api/upload ───────────────────────────────────────────

router.post("/upload", upload.single("pdf"), async (req: Request, res: Response): Promise<void> => {
  try {
    if (!req.file) {
      res.status(400).json({ error: "No PDF file uploaded" });
      return;
    }

    console.log(`\n📄 Received PDF: ${req.file.originalname} (${(req.file.size / 1024).toFixed(1)}KB)`);

    // Step 1-3: Extract text, recursive chunking, attach metadata
    console.log(`🧠 Processing with smart recursive chunking...`);
    const { chunks, totalChars } = await processDocument(req.file.path, 500, 50);

    if (chunks.length === 0) {
      res.status(400).json({
        error: "Could not extract text from PDF. The file may be image-based or empty.",
      });
      return;
    }

    console.log(`📝 Extracted ${totalChars.toLocaleString()} characters`);
    console.log(`✂️  Smart-split into ${chunks.length} chunks (recursive: paragraphs → sentences → fixed)`);

    // Log chunking stats
    const chunkLengths = chunks.map((c) => c.text.length);
    const avgLen = Math.round(chunkLengths.reduce((a, b) => a + b, 0) / chunkLengths.length);
    const minLen = Math.min(...chunkLengths);
    const maxLen = Math.max(...chunkLengths);
    console.log(`📊 Chunk sizes — avg: ${avgLen}, min: ${minLen}, max: ${maxLen} chars`);

    // Count unique sections detected
    const sections = new Set(chunks.map((c) => c.metadata.sectionHeading).filter(Boolean));
    if (sections.size > 0) {
      console.log(`📑 Detected ${sections.size} section heading(s)`);
    }

    // Count pages spanned
    const pages = new Set(chunks.map((c) => c.metadata.pageNumber));
    console.log(`📖 Chunks span ${pages.size} page(s)`);

    // Step 4: Generate embeddings for all chunks
    console.log(`🧮 Generating embeddings for ${chunks.length} chunks...`);
    const chunkTexts = chunks.map((c) => c.text);
    const embeddings = await getEmbeddings(chunkTexts);

    // Step 5: Store chunks + embeddings + metadata
    clearStore();
    const entries = chunks.map((chunk, index) => ({
      text: chunk.text,
      embedding: embeddings[index],
      metadata: chunk.metadata,
    }));
    addToStore(entries);

    console.log(`✅ Stored ${getStoreSize()} chunks in vector store\n`);

    // Clean up uploaded file
    fs.unlinkSync(req.file.path);

    res.json({
      message: "PDF processed successfully",
      chunks: chunks.length,
      characterCount: totalChars,
      pages: pages.size,
      sections: [...sections],
      chunkingMethod: "recursive (paragraphs → sentences → fixed-size)",
    });
  } catch (error) {
    console.error("Upload error:", error);

    if (req.file && fs.existsSync(req.file.path)) {
      fs.unlinkSync(req.file.path);
    }

    if (error instanceof Error) {
      if (error.message.includes("Only PDF files")) {
        res.status(400).json({ error: error.message });
        return;
      }
      if (error.message.includes("too large") || error.message.includes("LIMIT_FILE_SIZE")) {
        res.status(400).json({ error: "File is too large. Maximum size is 10MB." });
        return;
      }
    }

    res.status(500).json({ error: "Failed to process PDF. Please try again." });
  }
});

// ── POST /api/ask ──────────────────────────────────────────────

router.post("/ask", async (req: Request, res: Response): Promise<void> => {
  try {
    const { question } = req.body;

    if (!question || typeof question !== "string" || question.trim().length === 0) {
      res.status(400).json({ error: "Please provide a question" });
      return;
    }

    if (getStoreSize() === 0) {
      res.status(400).json({ error: "No PDF has been uploaded yet. Please upload a PDF first." });
      return;
    }

    console.log(`\n❓ Question: "${question}"`);

    // Step 1: Embed the user's question
    const questionEmbedding = await getEmbedding(question);

    // Step 2: HYBRID SEARCH — combine vector + keyword search
    console.log(`🔀 Running hybrid search (vector + keyword)...`);
    const hybridResults = hybridSearch(questionEmbedding, question, getStore(), 10);

    const semanticCount = hybridResults.filter((r) => r.matchType === "semantic").length;
    const keywordCount = hybridResults.filter((r) => r.matchType === "keyword").length;
    const hybridCount = hybridResults.filter((r) => r.matchType === "hybrid").length;
    console.log(`   Found ${hybridResults.length} candidates (${semanticCount} semantic, ${keywordCount} keyword, ${hybridCount} hybrid)`);

    // Step 3: RE-RANK — score top candidates with multiple signals
    console.log(`🏆 Re-ranking top candidates...`);
    const topChunks = rerankResults(question, hybridResults, 3);

    console.log(`🔍 Final top ${topChunks.length} chunks:`);
    topChunks.forEach((chunk, i) => {
      const page = chunk.metadata?.pageNumber || "?";
      const section = chunk.metadata?.sectionHeading || "none";
      console.log(`   ${i + 1}. Page ${page} | ${section} | score: ${chunk.score.toFixed(4)} | via: ${chunk.matchType}`);
    });

    // Step 4: Send re-ranked chunks + question to Gemini
    const answer = await askQuestion(question, topChunks);

    console.log(`💬 Generated answer (${answer.length} chars)\n`);

    // Return answer with full metadata
    res.json({
      answer,
      sources: topChunks.map((chunk) => ({
        text: chunk.text,
        relevanceScore: parseFloat(chunk.score.toFixed(4)),
        pageNumber: chunk.metadata?.pageNumber || null,
        sectionHeading: chunk.metadata?.sectionHeading || null,
        matchType: chunk.matchType,
      })),
      searchInfo: {
        method: "hybrid (vector + keyword) with re-ranking",
        candidatesEvaluated: hybridResults.length,
        finalResults: topChunks.length,
      },
    });
  } catch (error) {
    console.error("Ask error:", error);

    if (error instanceof Error) {
      if (error.message.includes("API key") || error.message.includes("API_KEY")) {
        res.status(500).json({ error: "Gemini API key is invalid or missing. Check your .env file." });
        return;
      }
      if (error.message.includes("rate limit") || error.message.includes("429") || error.message.includes("RESOURCE_EXHAUSTED")) {
        res.status(429).json({ error: "Rate limited by Gemini. Please wait a moment and try again." });
        return;
      }
      if (error.message.includes("503") || error.message.includes("Service Unavailable") || error.message.includes("high demand")) {
        res.status(503).json({ error: "Gemini model is currently overloaded. Please try again in a few seconds." });
        return;
      }
    }

    res.status(500).json({ error: "Failed to process your question. Please try again." });
  }
});

export default router;
