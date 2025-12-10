import OpenAI from 'openai';
import { Agent, run, setDefaultOpenAIClient } from '@openai/agents';

const client = new OpenAI({
  apiKey: 'dummy-key',
  baseURL: 'http://localhost:8000/v1',
});
setDefaultOpenAIClient(client);

const agent = new Agent({
  name: 'test_agent',
  model: 'gpt-oss-20b',
  instructions: 'You are helpful.',
});

console.log('Testing SDK reasoning capture...\n');

const result = await run(agent, 'what is 2+2?', { stream: true });

for await (const event of result) {
  if (event.type === 'run_item_stream_event' && event.item?.type === 'reasoning_item') {
    console.log('=== REASONING_ITEM EVENT ===');
    console.log('Item type:', event.item.type);
    console.log('Raw item:', JSON.stringify(event.item.rawItem, null, 2));
    
    // Access summary text like SDK does internally
    const summary = event.item.rawItem?.summary;
    if (summary && summary[0]) {
      console.log('\nSUMMARY TEXT:', summary[0].text);
    }
  }
}

console.log('\n=== FINAL ===');
console.log('Output:', result.finalOutput);
