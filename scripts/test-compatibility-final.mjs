/**
 * Final OpenAI Agents SDK Compatibility Test for egemma
 * Tests: Tool calls, Reasoning/Thinking, Streaming
 */
import OpenAI from 'openai';
import { Agent, run, tool, setDefaultOpenAIClient } from '@openai/agents';
import { z } from 'zod';

const client = new OpenAI({
  apiKey: 'dummy-key',
  baseURL: 'http://localhost:8000/v1',
});
setDefaultOpenAIClient(client);

// Test tool
const getTimeTool = tool({
  name: 'get_time',
  description: 'Get current time',
  parameters: z.object({}),
  execute: async () => 'Current time: 15:45:00',
});

const agent = new Agent({
  name: 'test_agent',
  model: 'gpt-oss-20b',
  instructions: 'You are helpful. Use tools when needed.',
  tools: [getTimeTool],
});

console.log('========================================');
console.log('  OpenAI Agents SDK <-> egemma Test');
console.log('========================================\n');

const result = await run(agent, 'what time is it?', { stream: true });

let hasReasoning = false;
let reasoningText = '';
let hasToolCall = false;
let toolName = '';

for await (const event of result) {
  // Capture reasoning
  if (event.type === 'run_item_stream_event' && event.item?.type === 'reasoning_item') {
    hasReasoning = true;
    reasoningText = event.item.rawItem?.content?.[0]?.text || '';
    console.log('REASONING DETECTED');
    console.log('   "' + reasoningText.substring(0, 80) + '..."\n');
  }

  // Capture tool calls
  if (event.type === 'run_item_stream_event' && event.name === 'tool_called') {
    hasToolCall = true;
    toolName = event.item?.rawItem?.name || '';
    console.log('TOOL CALL DETECTED');
    console.log('   Tool: ' + toolName + '\n');
  }
}

console.log('========================================');
console.log('RESULTS:');
console.log('----------------------------------------');
console.log('  Reasoning/Thinking: ' + (hasReasoning ? 'PASS' : 'FAIL'));
console.log('  Tool Calls:         ' + (hasToolCall ? 'PASS' : 'FAIL'));
console.log('  Final Output:       ' + (result.finalOutput ? 'PASS' : 'FAIL'));
console.log('----------------------------------------');
console.log('  Output: "' + (result.finalOutput?.substring(0, 60) || '') + '"');
console.log('========================================');
