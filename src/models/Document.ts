import mongoose, { Schema, Document as MongoDoc } from "mongoose";

export interface IDocument extends MongoDoc {
  filename: string;
  fileSize: number;
  totalChars: number;
  totalChunks: number;
  totalPages: number;
  sections: string[];
  chunkingMethod: string;
  uploadedAt: Date;
}

const documentSchema = new Schema<IDocument>({
  filename: { type: String, required: true },
  fileSize: { type: Number, required: true },
  totalChars: { type: Number, required: true },
  totalChunks: { type: Number, required: true },
  totalPages: { type: Number, required: true },
  sections: { type: [String], default: [] },
  chunkingMethod: { type: String, default: "recursive" },
  uploadedAt: { type: Date, default: Date.now },
});

documentSchema.index({ uploadedAt: -1 });

export default mongoose.model<IDocument>("Document", documentSchema);
