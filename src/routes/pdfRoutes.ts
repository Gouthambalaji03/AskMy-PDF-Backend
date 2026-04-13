/**
 * pdfRoutes.ts — API Routes for PDF Upload, Query, and Chat
 *
 * This file connects all the pieces of our RAG pipeline:
 *
 *   POST /api/upload  → Upload a PDF, extract text, chunk it, embed it, store it
 *   POST /api/ask     → Ask a question, retrieve relevant chunks, generate answer
 *
 * The full RAG flow for a question:
 *   1. User asks "What is X?"
 *   2. We embed the question into a vector
 *   3. We compare it against all stored chunk vectors (cosine similarity)
 *   4. We take the top 3 most similar chunks
 *   5. We send those chunks + the question to GPT-4o-mini
 *   6. GPT-4o-mini answers based ONLY on those chunks
 *   7. We return the answer to the user
 */

import { Router, Request, Response } from "express";
import multer from "multer";
import path from "path";
import fs from "fs";
import { extractTextFromPDF, splitTextIntoChunks } from "../services/pdfService";
import { getEmbeddings, getEmbedding, findTopMatches } from "../services/embeddingService";
import { addToStore, clearStore, getStore, getStoreSize } from "../store/vectorStore";
import { askQuestion } from "../services/chatService";

const router = Router();

// Configure multer for file uploads
// Multer handles multipart/form-data (file uploads) in Express
const storage = multer.diskStorage({
  destination: (_req, _file, cb) => {
    const uploadDir = path.join(__dirname, "../../uploads");
    // Create uploads directory if it doesn't exist
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir, { recursive: true });
    }
    cb(null, uploadDir);
  },
  filename: (_req, file, cb) => {
    // Add timestamp to prevent filename collisions
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
    // Only allow PDF files
    if (file.mimetype === "application/pdf") {
      cb(null, true);
    } else {
      cb(new Error("Only PDF files are allowed"));
    }
  },
});

/**
 * POST /api/upload
 *
 * This endpoint handles the full "ingestion" pipeline:
 *   1. Receive the uploaded PDF file
 *   2. Extract all text from the PDF
 *   3. Split the text into ~500-character chunks with 50-char overlap
 *   4. Send all chunks to OpenAI to get embedding vectors
 *   5. Store each { text, embedding } pair in our in-memory vector store
 *
 * After this, the vector store is ready to answer questions!
 */
router.post("/upload", upload.single("pdf"), async (req: Request, res: Response): Promise<void> => {
  try {
    // Check that a file was actually uploaded
    if (!req.file) {
      res.status(400).json({ error: "No PDF file uploaded" });
      return;
    }

    console.log(`📄 Received PDF: ${req.file.originalname} (${(req.file.size / 1024).toFixed(1)}KB)`);

    // Step 1: Extract text from the PDF
    const text = await extractTextFromPDF(req.file.path);

    if (!text || text.trim().length === 0) {
      res.status(400).json({ error: "Could not extract text from PDF. The file may be image-based or empty." });
      return;
    }

    console.log(`📝 Extracted ${text.length} characters of text`);

    // Step 2: Split into chunks
    const chunks = splitTextIntoChunks(text, 500, 50);
    console.log(`✂️  Split into ${chunks.length} chunks`);

    // Step 3: Get embeddings for all chunks (batched API call)
    console.log(`🧮 Generating embeddings for ${chunks.length} chunks...`);
    const embeddings = await getEmbeddings(chunks);

    // Step 4: Store chunks + embeddings in our vector store
    // Clear previous data (we only support one PDF at a time for simplicity)
    clearStore();

    const entries = chunks.map((text, index) => ({
      text,
      embedding: embeddings[index],
    }));
    addToStore(entries);

    console.log(`✅ Stored ${getStoreSize()} chunks in vector store`);

    // Clean up the uploaded file (we've extracted what we need)
    fs.unlinkSync(req.file.path);

    res.json({
      message: "PDF processed successfully",
      chunks: chunks.length,
      characterCount: text.length,
    });
  } catch (error) {
    console.error("Upload error:", error);

    // Clean up file on error
    if (req.file && fs.existsSync(req.file.path)) {
      fs.unlinkSync(req.file.path);
    }

    // Provide user-friendly error messages
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

/**
 * POST /api/ask
 *
 * This is the full RAG query pipeline:
 *   1. Receive the user's question
 *   2. Convert the question into an embedding vector
 *   3. Find the top 3 most similar chunks (cosine similarity)
 *   4. Send those chunks + question to Gemini 2.0 Flash
 *   5. Return the AI-generated answer
 *
 * Request body: { "question": "What is photosynthesis?" }
 */
router.post("/ask", async (req: Request, res: Response): Promise<void> => {
  try {
    const { question } = req.body;

    // Validate that a question was provided
    if (!question || typeof question !== "string" || question.trim().length === 0) {
      res.status(400).json({ error: "Please provide a question" });
      return;
    }

    // Check that a PDF has been uploaded first
    if (getStoreSize() === 0) {
      res.status(400).json({ error: "No PDF has been uploaded yet. Please upload a PDF first." });
      return;
    }

    console.log(`❓ Question: "${question}"`);

    // Step 1: Embed the user's question
    const questionEmbedding = await getEmbedding(question);

    // Step 2: Find the top 3 most relevant chunks
    const topChunks = findTopMatches(questionEmbedding, getStore(), 3);

    console.log(`🔍 Found top ${topChunks.length} chunks (scores: ${topChunks.map((c) => c.score.toFixed(3)).join(", ")})`);

    // Step 3: Send chunks + question to Gemini for answer generation
    const answer = await askQuestion(question, topChunks);

    console.log(`💬 Generated answer (${answer.length} chars)`);

    // Return the answer along with the source chunks for transparency
    res.json({
      answer,
      sources: topChunks.map((chunk) => ({
        text: chunk.text,
        relevanceScore: parseFloat(chunk.score.toFixed(4)),
      })),
    });
  } catch (error) {
    console.error("Ask error:", error);

    // Handle Gemini-specific errors
    if (error instanceof Error) {
      if (error.message.includes("API key") || error.message.includes("API_KEY")) {
        res.status(500).json({ error: "Gemini API key is invalid or missing. Check your .env file." });
        return;
      }
      if (error.message.includes("rate limit") || error.message.includes("429") || error.message.includes("RESOURCE_EXHAUSTED")) {
        res.status(429).json({ error: "Rate limited by Gemini. Please wait a moment and try again." });
        return;
      }
    }

    res.status(500).json({ error: "Failed to process your question. Please try again." });
  }
});

export default router;
