/**
 * Test reasoning/thinking output from egemma
 * Tests if SDK receives reasoning_item_created events
 */
import OpenAI from 'openai';
import { Agent, run, setDefaultOpenAIClient } from '@openai/agents';

const client = new OpenAI({
  apiKey: 'dummy-key',
  baseURL: 'http://localhost:8000/v1',
});
setDefaultOpenAIClient(client);

// Simple agent without tools to force thinking
const agent = new Agent({
  name: 'test_agent',
  model: 'gpt-oss-20b',
  instructions: 'Think through problems step by step. Show your reasoning.',
  tools: [],
});

async function testReasoning() {
  console.log('=== Testing Reasoning/Thinking Output ===\n');
  console.log('Prompt: "What is 15 * 23? Think step by step."\n');

  try {
    const result = await run(agent, 'What is 15 * 23? Think step by step.', {
      stream: true,
    });

    let reasoningFound = false;
    let textFound = false;

    for await (const event of result) {
      // Log all event types
      console.log(`Event: ${event.type}`);

      // Check for reasoning items
      if (event.type === 'run_item_stream_event') {
        console.log(`  -> ${event.name} (item type: ${event.item?.type})`);

        if (event.name === 'reasoning_item_created') {
          reasoningFound = true;
          console.log('\n🤖 REASONING ITEM FOUND!');
          console.log('Raw item:', JSON.stringify(event.item?.rawItem, null, 2));
        }

        if (event.name === 'message_output_created') {
          textFound = true;
          console.log('\n📝 MESSAGE OUTPUT:');
          const content = event.item?.rawItem?.content;
          if (content) {
            for (const c of content) {
              if (c.type === 'output_text') {
                console.log(c.text?.substring(0, 200) + '...');
              }
            }
          }
        }
      }

      // Check raw model events for thinking field
      if (event.type === 'raw_model_stream_event') {
        const chunk = event.data;
        if (chunk?.choices?.[0]?.delta?.thinking) {
          console.log(`  [RAW THINKING]: ${chunk.choices[0].delta.thinking.substring(0, 50)}...`);
        }
      }
    }

    console.log('\n=== Summary ===');
    console.log(`Reasoning item found: ${reasoningFound ? '✅ YES' : '❌ NO'}`);
    console.log(`Text message found: ${textFound ? '✅ YES' : '❌ NO'}`);

    if (result.newItems) {
      console.log('\nNew items in result:');
      for (const item of result.newItems) {
        console.log(`  - ${item.type}`);
        if (item.type === 'reasoning_item') {
          console.log('    Content:', JSON.stringify(item.rawItem, null, 2).substring(0, 200));
        }
      }
    }

  } catch (error) {
    console.error('Error:', error.message);
  }
}

await testReasoning();
