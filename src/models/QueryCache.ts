import mongoose, { Schema, Document } from "mongoose";
import crypto from "crypto";

export interface IQueryCache extends Document {
  cacheKey: string;
  documentId: mongoose.Types.ObjectId;
  question: string;
  answer: string;
  sources: unknown;
  searchInfo: unknown;
  createdAt: Date;
}

const queryCacheSchema = new Schema<IQueryCache>({
  cacheKey: { type: String, required: true, unique: true },
  documentId: { type: Schema.Types.ObjectId, ref: "Document", required: true },
  question: { type: String, required: true },
  answer: { type: String, required: true },
  sources: { type: Schema.Types.Mixed, default: [] },
  searchInfo: { type: Schema.Types.Mixed, default: {} },
  createdAt: { type: Date, default: Date.now },
});

// TTL index — auto-delete cached queries after 24 hours
queryCacheSchema.index({ createdAt: 1 }, { expireAfterSeconds: 86400 });

export default mongoose.model<IQueryCache>("QueryCache", queryCacheSchema);

/** Generate a deterministic cache key from documentId + question */
export function generateCacheKey(documentId: string, question: string): string {
  const normalized = question.toLowerCase().trim().replace(/\s+/g, " ");
  return crypto.createHash("sha256").update(`${documentId}::${normalized}`).digest("hex");
}
