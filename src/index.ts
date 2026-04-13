/**
 * index.ts — Express Server Entry Point
 *
 * This is where our backend server starts. It:
 *   1. Loads environment variables from .env
 *   2. Sets up Express with CORS and JSON parsing
 *   3. Mounts our PDF routes under /api
 *   4. Starts listening on the configured port
 */

import dotenv from "dotenv";

// Load .env variables BEFORE importing anything that uses them (like Gemini)
dotenv.config();

import express from "express";
import cors from "cors";
import pdfRoutes from "./routes/pdfRoutes";

const app = express();
const PORT = process.env.PORT || 5000;
const allowedOrigins = (process.env.FRONTEND_URL || "http://localhost:3000")
  .split(",")
  .map((origin) => origin.trim().replace(/\/+$/, ""))
  .filter(Boolean);

// --- Middleware ---

// CORS: Allow the frontend (running on a different port) to call our API
// Without this, the browser would block requests from localhost:3000 to localhost:5000
app.use(
  cors({
    origin: (origin, callback) => {
      if (!origin) {
        callback(null, true);
        return;
      }
      const cleaned = origin.replace(/\/+$/, "");
      if (allowedOrigins.includes(cleaned)) {
        callback(null, origin);
        return;
      }
      callback(new Error(`CORS blocked for origin: ${origin}`));
    },
    methods: ["GET", "POST"],
  })
);

// Parse JSON request bodies (for the /api/ask endpoint)
app.use(express.json());

// --- Routes ---

// Mount all PDF-related routes under /api
app.use("/api", pdfRoutes);

// Health check endpoint (useful for deployment platforms like Railway)
app.get("/health", (_req, res) => {
  res.json({ status: "ok" });
});

// --- Start Server ---

app.listen(PORT, () => {
  console.log(`\n🚀 AskMyPDF backend running on http://localhost:${PORT}`);
  console.log(`📋 Endpoints:`);
  console.log(`   POST /api/upload  — Upload a PDF`);
  console.log(`   POST /api/ask     — Ask a question`);
  console.log(`   GET  /health      — Health check\n`);
  console.log(`🌐 Allowed frontend origins: ${allowedOrigins.join(", ")}`);

  // Warn if Gemini API key is not set
  if (!process.env.GEMINI_API_KEY) {
    console.warn("⚠️  WARNING: GEMINI_API_KEY is not set in .env file!");
    console.warn("   Copy .env.example to .env and add your API key.\n");
  }
});
