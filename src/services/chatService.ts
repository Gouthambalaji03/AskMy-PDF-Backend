/**
 * chatService.ts — LLM Chat Completion with Context (Google Gemini)
 *
 * =============================================================
 * RAG CONCEPT: What is "context window" and why RAG solves it
 * =============================================================
 *
 * Every LLM has a "context window" — the maximum amount of text
 * it can read and process in a single request. Think of it as
 * the AI's short-term memory:
 *
 *   - Gemini 2.0 Flash: ~1,000,000 tokens
 *   - GPT-4o-mini: ~128,000 tokens
 *   - Claude 3.5: ~200,000 tokens
 *
 * Even though 1M tokens sounds like a lot, there are problems
 * with just dumping an entire PDF into the context:
 *
 *   1. COST: You pay per token. Sending 100 pages when you only
 *      need 3 paragraphs wastes money on every single query.
 *
 *   2. ACCURACY: LLMs can get "lost" in very long contexts.
 *      They perform better when given focused, relevant info.
 *      (This is called the "lost in the middle" problem.)
 *
 *   3. SCALE: What if you have 50 PDFs? That's way more than
 *      any context window can handle.
 *
 * RAG solves this by RETRIEVING only the relevant chunks first,
 * then passing just those chunks (+ the question) to the LLM.
 * The LLM gets exactly the context it needs — nothing more.
 *
 * This is the "Augmented Generation" part of RAG:
 *   R = Retrieve relevant chunks (embedding + cosine similarity)
 *   A = Augment the prompt with those chunks
 *   G = Generate an answer using the LLM
 * =============================================================
 */

import { GoogleGenerativeAI } from "@google/generative-ai";

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY!);

// Get the chat model instance with a system instruction
const chatModel = genAI.getGenerativeModel({
  model: "gemini-2.5-flash",
  generationConfig: {
    temperature: 0.3, // Low temperature = more focused, less creative answers
    maxOutputTokens: 1000,
  },
});

/**
 * Sends the user's question + relevant context chunks to Gemini 2.0 Flash
 * and gets back an answer grounded in the provided context.
 *
 * The system prompt is crucial here: it tells the LLM to ONLY answer
 * based on the provided context. Without this instruction, the LLM
 * might "hallucinate" — make up information that sounds right but
 * isn't in the PDF.
 *
 * @param question - The user's question
 * @param contextChunks - The top relevant chunks retrieved from the vector store
 * @returns The AI-generated answer
 */
export async function askQuestion(
  question: string,
  contextChunks: { text: string; score: number }[]
): Promise<string> {
  // Format the context chunks into a numbered list for the LLM
  const contextText = contextChunks
    .map((chunk, index) => `[Source ${index + 1}]:\n${chunk.text}`)
    .join("\n\n");

  const systemPrompt = `You are a helpful assistant that answers questions based ONLY on the provided context from a PDF document.

Rules:
- Only use information from the context below to answer the question.
- If the context doesn't contain enough information to answer, say "I don't have enough information from the PDF to answer this question."
- Do NOT make up or infer information beyond what's explicitly stated in the context.
- Keep your answers concise and well-structured.
- If relevant, mention which source section the information came from.

Context from the PDF:
${contextText}`;

  // Combine the system instruction and user question into a single prompt
  const fullPrompt = `${systemPrompt}\n\nUser question: ${question}`;

  const result = await chatModel.generateContent(fullPrompt);
  const response = result.response;

  return response.text() || "Sorry, I could not generate an answer.";
}
