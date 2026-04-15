import mongoose, { Schema, Document } from "mongoose";

export interface IChunk extends Document {
  documentId: mongoose.Types.ObjectId;
  text: string;
  embedding: number[];
  pageNumber: number;
  sectionHeading: string | null;
  chunkIndex: number;
}

const chunkSchema = new Schema<IChunk>({
  documentId: { type: Schema.Types.ObjectId, ref: "Document", required: true, index: true },
  text: { type: String, required: true },
  embedding: { type: Schema.Types.Mixed, required: true }, // Mixed avoids casting 3072 floats
  pageNumber: { type: Number, required: true },
  sectionHeading: { type: String, default: null },
  chunkIndex: { type: Number, required: true },
});

chunkSchema.index({ documentId: 1, chunkIndex: 1 });

export default mongoose.model<IChunk>("Chunk", chunkSchema);
