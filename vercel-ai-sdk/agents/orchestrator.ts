require('dotenv').config();

import { generateObject } from 'ai';
import { z } from 'zod';
import { createGroq } from '@ai-sdk/groq';

const groq = createGroq({
  baseURL: 'https://api.groq.com/openai/v1',
  apiKey: process.env.GROQ_API_KEY || "",
});

async function implementTask(taskRequest: string) {
  const { object: taskPlan } = await generateObject({
    model: groq('llama-3.3-70b-versatile'),
    schema: z.object({
      tasks: z.array(
        z.object({
          purpose: z.string(),
          taskName: z.string(),
          changeType: z.enum(['create', 'modify', 'delete']),
        })
      ),
      estimatedEffort: z.enum(['low', 'medium', 'high']),
    }),
    system: 'You are a Project Manager responsible for designing an efficient task execution strategy.',
    prompt: `Create a work plan for the following task:
    ${taskRequest}`,
  });

  const taskChanges = await Promise.all(
    taskPlan.tasks.map(async (task) => {
      // Determine job roles based on task type
      const workerSystemPrompt = {
        create: {
          'Audience research': 'You are a Business Analyst. You are responsible for conducting in-depth research on the target audience.',
          'Content creation': 'You are a Content Strategist. You design engaging content strategies tailored to the audience.',
          'Account management': 'You are a Social Media Manager. You manage and optimize social media accounts.',
          'Performance analysis': 'You are a Marketing Analyst. You analyze data and measure the success of marketing strategies.',
        }[task.taskName] || 'You are an expert professional in this field.',
        modify: {
          'Account management': 'You are a Social Media Manager. You improve account management strategies to be more effective.',
        }[task.taskName] || 'You are a specialist enhancing task efficiency.',
        delete: 'You are an Operations Manager. You identify unnecessary tasks and remove them efficiently.',
      }[task.changeType];

      const { object: change } = await generateObject({
        model: groq('llama-3.3-70b-versatile'),
        schema: z.object({
          explanation: z.string(),
          actionItems: z.array(z.string()),
        }),
        system: workerSystemPrompt,
        prompt: `Implement changes for the following task:
        - ${task.taskName}
        
        Purpose of change: ${task.purpose}
        
        Explain the reason for the change and provide a list of necessary action items.`,
      });

      return {
        task,
        implementation: change,
      };
    })
  );

  // Modify Output for Clarity
  console.log('===== TASK PLAN =====');
  console.log(JSON.stringify(taskPlan, null, 2));

  console.log('\n===== TASK CHANGES =====');
  taskChanges.forEach((change, index) => {
    console.log(`\n${index + 1}. ${change.task.taskName}`);
    console.log(`   Purpose       : ${change.task.purpose}`);
    console.log(`   Change Type   : ${change.task.changeType}`);
    console.log(`   Explanation   : ${change.implementation.explanation}`);
    console.log(`   Action Items  :`);
    change.implementation.actionItems.forEach((item, idx) => {
      console.log(`     - ${item}`);
    });
  });

  return {
    plan: taskPlan,
    changes: taskChanges,
  };
}

// Example function call with a general task
const taskRequest = 'Develop a social media marketing strategy for a small business';
implementTask(taskRequest).then((result) => {
  console.log('\n===== TASK IMPLEMENTATION COMPLETED =====');
}).catch((error) => {
  console.error('Error:', error);
});