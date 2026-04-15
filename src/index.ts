import dotenv from "dotenv";
dotenv.config();

import express from "express";
import cors from "cors";
import mongoose from "mongoose";
import rateLimit from "express-rate-limit";
import { connectDB } from "./config/database";
import pdfRoutes from "./routes/pdfRoutes";

const app = express();
const PORT = process.env.PORT || 5000;
const allowedOrigins = (process.env.FRONTEND_URL || "http://localhost:3000")
  .split(",")
  .map((origin) => origin.trim().replace(/\/+$/, ""))
  .filter(Boolean);

// --- Middleware ---

app.use(
  cors({
    origin: (origin, callback) => {
      if (!origin) { callback(null, true); return; }
      const cleaned = origin.replace(/\/+$/, "");
      if (allowedOrigins.includes(cleaned)) { callback(null, origin); return; }
      callback(new Error(`CORS blocked for origin: ${origin}`));
    },
    methods: ["GET", "POST", "DELETE"],
  })
);

app.use(express.json());

// --- Rate Limiting ---

const askLimiter = rateLimit({
  windowMs: 60 * 1000, // 1 minute
  max: 15, // 15 requests per minute per IP
  message: { error: "Too many requests. Please wait a moment and try again." },
  standardHeaders: true,
  legacyHeaders: false,
});

const uploadLimiter = rateLimit({
  windowMs: 60 * 1000,
  max: 5, // 5 uploads per minute per IP
  message: { error: "Too many uploads. Please wait a moment." },
  standardHeaders: true,
  legacyHeaders: false,
});

app.use("/api/ask", askLimiter);
app.use("/api/upload", uploadLimiter);

// --- Routes ---

app.use("/api", pdfRoutes);

app.get("/health", (_req, res) => {
  const dbState = mongoose.connection.readyState;
  const dbStatus = dbState === 1 ? "connected" : dbState === 2 ? "connecting" : "disconnected";
  res.json({ status: "ok", database: dbStatus });
});

// --- Start Server (connect DB first) ---

connectDB().then(() => {
  app.listen(PORT, () => {
    console.log(`\n🚀 AskMyPDF backend running on http://localhost:${PORT}`);
    console.log(`📋 Endpoints:`);
    console.log(`   POST   /api/upload            — Upload a PDF`);
    console.log(`   POST   /api/ask               — Ask a question`);
    console.log(`   POST   /api/ask/stream        — Ask with streaming (SSE)`);
    console.log(`   GET    /api/documents          — List all documents`);
    console.log(`   GET    /api/documents/:id/chat — Chat history`);
    console.log(`   DELETE /api/documents/:id      — Delete a document`);
    console.log(`   GET    /health                 — Health check\n`);
    console.log(`🛡️  Rate limits: 15 asks/min, 5 uploads/min per IP`);
    console.log(`🌐 Allowed origins: ${allowedOrigins.join(", ")}`);
  });
});
