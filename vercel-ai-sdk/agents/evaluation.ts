require('dotenv').config();

import { generateObject, generateText } from 'ai';
import { z } from 'zod';
import { createGroq } from '@ai-sdk/groq';

const groq = createGroq({
  baseURL: 'https://api.groq.com/openai/v1',
  apiKey: process.env.GROQ_API_KEY || "",
});

async function writeArticleWithFeedback(topic: string) {
  let currentArticle = '';
  let iterations = 0;
  const MAX_ITERATIONS = 3;

  // Initial article generation
  const { text: article } = await generateText({
    model: groq('llama-3.1-8b-instant'),
    system: 'You are a writer. Your task is to write a concise article in only 6 sentences! You might get additional feedback from your supervisor!',
    prompt: `Write a 6-sentence article on the topic: ${topic}`,
  });

  currentArticle = article;

  // Evaluation-optimization loop
  while (iterations < MAX_ITERATIONS) {
    // Evaluate current article
    const { object: evaluation } = await generateObject({
      model: groq('llama-3.3-70b-versatile'), // use a larger model to evaluate
      schema: z.object({
        qualityScore: z.number().min(1).max(10),
        clearAndConcise: z.boolean(),
        engaging: z.boolean(),
        informative: z.boolean(),
        specificIssues: z.array(z.string()),
        improvementSuggestions: z.array(z.string()),
      }),
      system: "You are a writing supervisor! Your agency specializes in concise articles! Your task is to evaluate the given article and provide feedback for improvements! Repeat until the article meets your requirements!",
      prompt: `Evaluate this article:

      Article: ${currentArticle}

      Consider:
      1. Overall quality
      2. Clarity and conciseness
      3. Engagement level
      4. Informative value`,
    });

    // Check if quality meets threshold
    if (
      evaluation.qualityScore >= 8 &&
      evaluation.clearAndConcise &&
      evaluation.engaging &&
      evaluation.informative
    ) {
      break;
    }

    // Generate improved article based on feedback
    const { text: improvedArticle } = await generateText({
      model: groq('llama-3.3-70b-versatile'), // use a larger model
      system: 'You are an expert article writer.',
      prompt: `Improve this article based on the following feedback:
      ${evaluation.specificIssues.join('\n')}
      ${evaluation.improvementSuggestions.join('\n')}

      Current Article: ${currentArticle}`,
    });

    currentArticle = improvedArticle;
    iterations++;
  }

  return {
    finalArticle: currentArticle,
    iterationsRequired: iterations,
  };
}

// Execute the function and print the output
(async () => {
  const topic = 'Machine learning in agriculture'; // You can change this topic as needed
  const result = await writeArticleWithFeedback(topic);
  console.log(`Final Article:\n${result.finalArticle}`);
  console.log(`Iterations Required: ${result.iterationsRequired}`);
})();