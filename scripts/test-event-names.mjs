/**
 * Debug: Check exact event structure from SDK
 */
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

console.log('Checking event structure for reasoning...\n');

const result = await run(agent, 'what is 2+2?', { stream: true });

for await (const event of result) {
  if (event.type === 'run_item_stream_event') {
    console.log('run_item_stream_event:');
    console.log('  name:', event.name);
    console.log('  item.type:', event.item?.type);

    if (event.item?.type === 'reasoning_item') {
      console.log('  >>> REASONING ITEM <<<');
      console.log('  rawItem.type:', event.item?.rawItem?.type);
      console.log('  rawItem.content[0].type:', event.item?.rawItem?.content?.[0]?.type);
      console.log('  rawItem.content[0].text:', event.item?.rawItem?.content?.[0]?.text?.substring(0, 50));
    }
    console.log('');
  }
}
