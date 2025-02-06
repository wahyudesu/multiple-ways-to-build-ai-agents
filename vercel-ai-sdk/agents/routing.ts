import { generateObject, generateText } from 'ai';
import { z } from 'zod';

require('dotenv').config();

import { createGroq } from '@ai-sdk/groq';

const groq = createGroq({
  baseURL: 'https://api.groq.com/openai/v1', // Change if using a proxy
  apiKey: process.env.GROQ_API_KEY || "",
});

async function handleCustomerQuery(query: string) {
  const model = groq('llama-3.3-70b-versatile');

  // Step 1: Classify the query type
  const { object: classification } = await generateObject({
    model,
    schema: z.object({
      reasoning: z.string(),
      type: z.enum(['general', 'refund', 'technical']),
      complexity: z.enum(['simple', 'complex']),
    }),
    prompt: `Classify this customer query:
    ${query}

    Determine:
    1. Query type (general, refund, or technical)
    2. Complexity (simple or complex)
    3. Brief reasoning for classification`,
  });

  // Route based on classification
  // Determine the model and system prompt based on query type and complexity
  const { text: response } = await generateText({
    model:
      classification.complexity === 'simple'
        ? groq('llama3-8b-8192')
        : groq('llama-3.1-8b-instant'),
    system: {
      general:
        'You are an expert customer service agent handling general inquiries.',
      refund:
        'You are a customer service agent specializing in refund requests. Follow company policies and gather necessary information.',
      technical:
        'You are a technical support specialist with in-depth knowledge of the product. Focus on clear, step-by-step troubleshooting.',
    }[classification.type],
    prompt: query,
  });

  console.log('Response:', response);
  console.log('Classification:', classification);
}

const query = "I am experiencing a 404 error when trying to log in to my app. How can I fix this?";
handleCustomerQuery(query);