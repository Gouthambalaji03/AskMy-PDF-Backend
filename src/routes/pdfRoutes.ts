/**
 * pdfRoutes.ts — API Routes with MongoDB, Multi-Doc, Caching, Chat History
 *
 * Endpoints:
 *   POST   /api/upload            — Upload & process a PDF
 *   POST   /api/ask               — Ask a question (with caching)
 *   GET    /api/documents         — List all uploaded documents
 *   GET    /api/documents/:id/chat — Get chat history for a document
 *   DELETE /api/documents/:id     — Delete a document and all its data
 */

import { Router, Request, Response } from "express";
import multer from "multer";
import path from "path";
import fs from "fs";
import { processDocument } from "../services/pdfService";
import { getEmbeddings, getEmbedding, rerankResults } from "../services/embeddingService";
import { addChunks, getChunkCount, deleteChunksByDocument, hybridSearchMongo } from "../store/vectorStore";
import { askQuestion, askQuestionStream, ChatTurn } from "../services/chatService";
import DocumentModel from "../models/Document";
import Chat from "../models/Chat";
import QueryCache, { generateCacheKey } from "../models/QueryCache";

// ── Constants ─────────────────────────────────────────────────

const MAX_QUESTION_LENGTH = 1000;
const CHAT_HISTORY_LIMIT = 5; // last N Q&A pairs sent as conversational memory

const router = Router();

// ── Multer Config ──────────────────────────────────────────────

const storage = multer.diskStorage({
  destination: (_req, _file, cb) => {
    const uploadDir = path.join(__dirname, "../../uploads");
    if (!fs.existsSync(uploadDir)) fs.mkdirSync(uploadDir, { recursive: true });
    cb(null, uploadDir);
  },
  filename: (_req, file, cb) => {
    cb(null, `${Date.now()}-${file.originalname}`);
  },
});

const upload = multer({
  storage,
  limits: { fileSize: parseInt(process.env.MAX_FILE_SIZE || "10485760") },
  fileFilter: (_req, file, cb) => {
    if (file.mimetype === "application/pdf") cb(null, true);
    else cb(new Error("Only PDF files are allowed"));
  },
});

// ── POST /api/upload ───────────────────────────────────────────

router.post("/upload", upload.single("pdf"), async (req: Request, res: Response): Promise<void> => {
  try {
    if (!req.file) { res.status(400).json({ error: "No PDF file uploaded" }); return; }

    console.log(`\n📄 Received: ${req.file.originalname} (${(req.file.size / 1024).toFixed(1)}KB)`);

    // 1. Extract + smart chunk
    const { chunks, totalChars } = await processDocument(req.file.path, 500, 50);
    if (chunks.length === 0) {
      res.status(400).json({ error: "Could not extract text from PDF." });
      return;
    }

    const sections = [...new Set(chunks.map((c) => c.metadata.sectionHeading).filter(Boolean))] as string[];
    const pages = new Set(chunks.map((c) => c.metadata.pageNumber));

    console.log(`📝 ${totalChars.toLocaleString()} chars, ${chunks.length} chunks, ${pages.size} pages`);

    // 2. Generate embeddings
    console.log(`🧮 Generating embeddings for ${chunks.length} chunks...`);
    const embeddings = await getEmbeddings(chunks.map((c) => c.text));

    // 3. Save document record to MongoDB
    const doc = await DocumentModel.create({
      filename: req.file.originalname,
      fileSize: req.file.size,
      totalChars,
      totalChunks: chunks.length,
      totalPages: pages.size,
      sections,
      chunkingMethod: "recursive (paragraphs → sentences → fixed-size)",
    });

    // 4. Save chunks + embeddings to MongoDB
    const entries = chunks.map((chunk, i) => ({
      text: chunk.text,
      embedding: embeddings[i],
      metadata: chunk.metadata,
    }));
    await addChunks(doc._id.toString(), entries);

    console.log(`✅ Stored ${await getChunkCount(doc._id.toString())} chunks in MongoDB\n`);

    // Clean up file
    fs.unlinkSync(req.file.path);

    res.json({
      message: "PDF processed successfully",
      documentId: doc._id,
      filename: doc.filename,
      chunks: chunks.length,
      characterCount: totalChars,
      pages: pages.size,
      sections,
      chunkingMethod: doc.chunkingMethod,
    });
  } catch (error) {
    console.error("Upload error:", error);
    if (req.file && fs.existsSync(req.file.path)) fs.unlinkSync(req.file.path);

    if (error instanceof Error) {
      if (error.message.includes("Only PDF files")) { res.status(400).json({ error: error.message }); return; }
      if (error.message.includes("LIMIT_FILE_SIZE")) { res.status(400).json({ error: "File too large. Max 10MB." }); return; }
    }
    res.status(500).json({ error: "Failed to process PDF." });
  }
});

// ── Shared: Validate + resolve question request ──────────────

async function validateAskRequest(req: Request, res: Response) {
  const { question, documentId } = req.body;

  if (!question || typeof question !== "string" || question.trim().length === 0) {
    res.status(400).json({ error: "Please provide a question." });
    return null;
  }

  if (question.trim().length > MAX_QUESTION_LENGTH) {
    res.status(400).json({ error: `Question too long. Maximum ${MAX_QUESTION_LENGTH} characters.` });
    return null;
  }

  let docId = documentId;
  if (!docId) {
    const latest = await DocumentModel.findOne().sort({ uploadedAt: -1 }).lean();
    if (!latest) { res.status(400).json({ error: "No PDF uploaded yet." }); return null; }
    docId = latest._id.toString();
  }

  // Validate ObjectId format
  if (!/^[a-f\d]{24}$/i.test(docId)) {
    res.status(400).json({ error: "Invalid document ID format." });
    return null;
  }

  const doc = await DocumentModel.findById(docId).lean();
  if (!doc) { res.status(404).json({ error: "Document not found." }); return null; }

  return { question: question.trim(), docId, doc };
}

// ── Shared: Load chat history for conversational memory ───────

async function loadChatMemory(docId: string): Promise<ChatTurn[]> {
  const recentChats = await Chat.find({ documentId: docId })
    .sort({ createdAt: -1 })
    .limit(CHAT_HISTORY_LIMIT)
    .select("question answer")
    .lean();

  // Reverse so oldest is first (chronological order)
  return recentChats.reverse().map((c) => ({
    question: c.question,
    answer: c.answer.slice(0, 300), // Truncate long answers to save tokens
  }));
}

// ── Shared: Handle ask errors ─────────────────────────────────

function handleAskError(error: unknown, res: Response) {
  console.error("Ask error:", error);

  if (error instanceof Error) {
    if (error.message === "QUOTA_EXHAUSTED") {
      res.status(429).json({ error: "Gemini API daily quota exhausted. Try again later." });
      return;
    }
    if (error.message.includes("API key") || error.message.includes("API_KEY")) {
      res.status(500).json({ error: "Gemini API key is invalid or missing." });
      return;
    }
    if (error.message.includes("429") || error.message.includes("RESOURCE_EXHAUSTED")) {
      res.status(429).json({ error: "Rate limited by Gemini. Wait a moment and try again." });
      return;
    }
    if (error.message.includes("503") || error.message.includes("Service Unavailable")) {
      res.status(503).json({ error: "Gemini model overloaded. Try again shortly." });
      return;
    }
  }
  res.status(500).json({ error: "Failed to process your question." });
}

// ── POST /api/ask (non-streaming, used for cache hits) ────────

router.post("/ask", async (req: Request, res: Response): Promise<void> => {
  try {
    const validated = await validateAskRequest(req, res);
    if (!validated) return;
    const { question, docId, doc } = validated;

    console.log(`\n❓ Question: "${question}" (doc: ${doc.filename})`);

    // ── Check query cache ──────────────────────────────────────
    const cacheKey = generateCacheKey(docId, question);
    const cached = await QueryCache.findOne({ cacheKey }).lean();

    if (cached) {
      console.log(`⚡ Cache HIT — returning cached answer`);

      const cachedSources = cached.sources as Record<string, unknown>[];
      const cachedSearchInfo = cached.searchInfo as Record<string, unknown>;

      await Chat.create({
        documentId: docId, question, answer: cached.answer,
        sources: cachedSources, searchInfo: cachedSearchInfo, cached: true,
      });

      res.json({ answer: cached.answer, sources: cachedSources, searchInfo: cachedSearchInfo, cached: true });
      return;
    }

    // ── Cache MISS — run full RAG pipeline ─────────────────────
    console.log(`🔍 Cache MISS — running hybrid search...`);

    const chatHistory = await loadChatMemory(docId);
    if (chatHistory.length > 0) console.log(`💬 Conversational memory: ${chatHistory.length} previous turns`);

    const questionEmbedding = await getEmbedding(question);
    const hybridResults = await hybridSearchMongo(questionEmbedding, question, docId, 10);
    console.log(`   Found ${hybridResults.length} candidates`);

    const topChunks = rerankResults(question, hybridResults, 3);
    topChunks.forEach((c, i) => {
      console.log(`   ${i + 1}. Page ${c.metadata?.pageNumber || "?"} | score: ${c.score.toFixed(4)} | ${c.matchType}`);
    });

    const answer = await askQuestion(question, topChunks, chatHistory);
    console.log(`💬 Answer generated (${answer.length} chars)\n`);

    const sources = topChunks.map((c) => ({
      text: c.text,
      relevanceScore: parseFloat(c.score.toFixed(4)),
      pageNumber: c.metadata?.pageNumber || null,
      sectionHeading: c.metadata?.sectionHeading || null,
      matchType: c.matchType,
    }));

    const searchInfo = {
      method: "hybrid (vector + keyword) with re-ranking",
      candidatesEvaluated: hybridResults.length,
      finalResults: topChunks.length,
    };

    await Chat.create({ documentId: docId, question, answer, sources, searchInfo, cached: false });

    try {
      await QueryCache.create({ cacheKey, documentId: docId, question, answer, sources, searchInfo });
    } catch (cacheErr: unknown) {
      const msg = cacheErr instanceof Error ? cacheErr.message : "";
      if (!msg.includes("duplicate key") && !msg.includes("11000")) console.error("Cache write error:", cacheErr);
    }

    res.json({ answer, sources, searchInfo, cached: false });
  } catch (error) {
    handleAskError(error, res);
  }
});

// ── POST /api/ask/stream (SSE streaming response) ─────────────

router.post("/ask/stream", async (req: Request, res: Response): Promise<void> => {
  let sseStarted = false; // Track whether SSE headers have been sent
  let clientDisconnected = false; // Track client disconnect

  // Detect client disconnect mid-stream
  req.on("close", () => {
    clientDisconnected = true;
    console.log("   Client disconnected mid-stream");
  });

  try {
    const validated = await validateAskRequest(req, res);
    if (!validated) return;
    const { question, docId, doc } = validated;

    console.log(`\n❓ [STREAM] Question: "${question}" (doc: ${doc.filename})`);

    // ── Check query cache — return full answer as single SSE ───
    const cacheKey = generateCacheKey(docId, question);
    const cached = await QueryCache.findOne({ cacheKey }).lean();

    if (cached) {
      console.log(`⚡ Cache HIT — streaming cached answer`);

      const cachedSources = cached.sources as Record<string, unknown>[];
      const cachedSearchInfo = cached.searchInfo as Record<string, unknown>;

      await Chat.create({
        documentId: docId, question, answer: cached.answer,
        sources: cachedSources, searchInfo: cachedSearchInfo, cached: true,
      });

      res.writeHead(200, {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        Connection: "keep-alive",
        "Access-Control-Allow-Origin": req.headers.origin || "*",
      });
      sseStarted = true;

      res.write(`data: ${JSON.stringify({ type: "chunk", text: cached.answer })}\n\n`);
      res.write(`data: ${JSON.stringify({ type: "done", sources: cachedSources, searchInfo: cachedSearchInfo, cached: true })}\n\n`);
      res.end();
      return;
    }

    // ── Cache MISS — stream from LLM ──────────────────────────
    console.log(`🔍 Cache MISS — running hybrid search...`);

    const chatHistory = await loadChatMemory(docId);
    if (chatHistory.length > 0) console.log(`💬 Conversational memory: ${chatHistory.length} previous turns`);

    const questionEmbedding = await getEmbedding(question);
    const hybridResults = await hybridSearchMongo(questionEmbedding, question, docId, 10);
    console.log(`   Found ${hybridResults.length} candidates`);

    const topChunks = rerankResults(question, hybridResults, 3);
    topChunks.forEach((c, i) => {
      console.log(`   ${i + 1}. Page ${c.metadata?.pageNumber || "?"} | score: ${c.score.toFixed(4)} | ${c.matchType}`);
    });

    const sources = topChunks.map((c) => ({
      text: c.text,
      relevanceScore: parseFloat(c.score.toFixed(4)),
      pageNumber: c.metadata?.pageNumber || null,
      sectionHeading: c.metadata?.sectionHeading || null,
      matchType: c.matchType,
    }));

    const searchInfo = {
      method: "hybrid (vector + keyword) with re-ranking",
      candidatesEvaluated: hybridResults.length,
      finalResults: topChunks.length,
    };

    // SSE headers
    res.writeHead(200, {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
      "Access-Control-Allow-Origin": req.headers.origin || "*",
    });
    sseStarted = true;

    // Stream LLM tokens (skip writes if client disconnected)
    const fullAnswer = await askQuestionStream(question, topChunks, chatHistory, (text) => {
      if (!clientDisconnected) {
        res.write(`data: ${JSON.stringify({ type: "chunk", text })}\n\n`);
      }
    });

    console.log(`💬 Streamed answer (${fullAnswer.length} chars)\n`);

    if (!clientDisconnected) {
      res.write(`data: ${JSON.stringify({ type: "done", sources, searchInfo, cached: false })}\n\n`);
      res.end();
    }

    // Save to chat history + cache regardless of disconnect
    await Chat.create({ documentId: docId, question, answer: fullAnswer, sources, searchInfo, cached: false });
    try {
      await QueryCache.create({ cacheKey, documentId: docId, question, answer: fullAnswer, sources, searchInfo });
    } catch (cacheErr: unknown) {
      const msg = cacheErr instanceof Error ? cacheErr.message : "";
      if (!msg.includes("duplicate key") && !msg.includes("11000")) console.error("Cache write error:", cacheErr);
    }
  } catch (error) {
    if (sseStarted) {
      // Headers already sent — can't use res.status(). Send SSE error event instead.
      const errMsg = error instanceof Error ? error.message : "Something went wrong";
      console.error("Stream error (SSE already started):", errMsg);
      if (!clientDisconnected) {
        res.write(`data: ${JSON.stringify({ type: "error", error: errMsg })}\n\n`);
        res.end();
      }
    } else {
      // Headers not sent yet — safe to use normal JSON error response
      handleAskError(error, res);
    }
  }
});

// ── GET /api/documents ─────────────────────────────────────────

router.get("/documents", async (_req: Request, res: Response): Promise<void> => {
  try {
    const documents = await DocumentModel.find()
      .sort({ uploadedAt: -1 })
      .select("filename fileSize totalChunks totalPages sections uploadedAt")
      .lean();

    res.json({ documents });
  } catch (error) {
    console.error("List documents error:", error);
    res.status(500).json({ error: "Failed to list documents." });
  }
});

// ── GET /api/documents/:id/chat ────────────────────────────────

router.get("/documents/:id/chat", async (req: Request, res: Response): Promise<void> => {
  try {
    const { id } = req.params;

    const doc = await DocumentModel.findById(id).lean();
    if (!doc) { res.status(404).json({ error: "Document not found." }); return; }

    const messages = await Chat.find({ documentId: id })
      .sort({ createdAt: 1 })
      .select("question answer sources searchInfo cached createdAt")
      .lean();

    res.json({ documentId: id, filename: doc.filename, messages });
  } catch (error) {
    console.error("Chat history error:", error);
    res.status(500).json({ error: "Failed to load chat history." });
  }
});

// ── DELETE /api/documents/:id ──────────────────────────────────

router.delete("/documents/:id", async (req: Request, res: Response): Promise<void> => {
  try {
    const id = req.params.id as string;

    const doc = await DocumentModel.findById(id).lean();
    if (!doc) { res.status(404).json({ error: "Document not found." }); return; }

    // Delete everything related to this document
    await Promise.all([
      DocumentModel.deleteOne({ _id: id }),
      deleteChunksByDocument(id),
      Chat.deleteMany({ documentId: id }),
      QueryCache.deleteMany({ documentId: id }),
    ]);

    console.log(`🗑️  Deleted document "${doc.filename}" and all related data`);
    res.json({ message: `Deleted "${doc.filename}" successfully.` });
  } catch (error) {
    console.error("Delete error:", error);
    res.status(500).json({ error: "Failed to delete document." });
  }
});

export default router;
