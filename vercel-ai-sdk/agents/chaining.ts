require('dotenv').config();

import { generateObject, generateText } from 'ai';
import { z } from 'zod';
import { createGroq } from '@ai-sdk/groq';

const groq = createGroq({
  baseURL: 'https://api.groq.com/openai/v1',
  apiKey: process.env.GROQ_API_KEY || "",
});

async function generateMarketingCopy(input: string) {
  const model = groq('llama-3.3-70b-versatile');

  console.log("Generating initial marketing copy...");
  const { text: copy } = await generateText({
    model,
    prompt: `Write persuasive marketing copy for: ${input}. Focus on benefits and emotional appeal.`,
  });

  console.log("Evaluating marketing copy quality...");
  const { object: qualityMetrics } = await generateObject({
    model,
    schema: z.object({
      hasCallToAction: z.boolean(),
      emotionalAppeal: z.number().min(1).max(10),
      clarity: z.number().min(1).max(10),
    }),
    prompt: `Evaluate this marketing copy for:
    1. Presence of call to action (true/false)
    2. Emotional appeal (1-10)
    3. Clarity (1-10)
    
    Copy to evaluate: ${copy}`,
  });

  if (
    !qualityMetrics.hasCallToAction ||
    qualityMetrics.emotionalAppeal < 7 ||
    qualityMetrics.clarity < 7
  ) {
    console.log("Improving marketing copy based on quality evaluation...");
    const { text: improvedCopy } = await generateText({
      model,
      prompt: `Rewrite this marketing copy with:
      ${!qualityMetrics.hasCallToAction ? '- A clear call to action' : ''}
      ${qualityMetrics.emotionalAppeal < 7 ? '- Stronger emotional appeal' : ''}
      ${qualityMetrics.clarity < 7 ? '- Improved clarity and directness' : ''}
      
      Original copy: ${copy}`,
    });

    console.log("\nFinal Improved Marketing Copy:\n", improvedCopy);
    console.log("\nQuality Metrics:", qualityMetrics);
    return;
  }

  console.log("\nFinal Marketing Copy:\n", copy);
  console.log("\nQuality Metrics:", qualityMetrics);
}

const input = "A revolutionary AI-powered chatbot for businesses.";

generateMarketingCopy(input).catch(console.error);
