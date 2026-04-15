import mongoose, { Schema, Document } from "mongoose";

export interface IChat extends Document {
  documentId: mongoose.Types.ObjectId;
  question: string;
  answer: string;
  sources: {
    text: string;
    score: number;
    pageNumber: number | null;
    sectionHeading: string | null;
    matchType: string;
  }[];
  searchInfo: {
    method: string;
    candidatesEvaluated: number;
    finalResults: number;
  };
  cached: boolean;
  createdAt: Date;
}

const chatSchema = new Schema<IChat>({
  documentId: { type: Schema.Types.ObjectId, ref: "Document", required: true },
  question: { type: String, required: true },
  answer: { type: String, required: true },
  sources: { type: Schema.Types.Mixed, default: [] },
  searchInfo: { type: Schema.Types.Mixed, default: {} },
  cached: { type: Boolean, default: false },
  createdAt: { type: Date, default: Date.now },
});

chatSchema.index({ documentId: 1, createdAt: 1 });

export default mongoose.model<IChat>("Chat", chatSchema);
