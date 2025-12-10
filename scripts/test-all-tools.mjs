/**
 * Comprehensive tool test for @openai/agents SDK against egemma
 * Tests all cognition-cli tools with natural language queries
 */
import OpenAI from 'openai';
import { Agent, run, tool, setDefaultOpenAIClient } from '@openai/agents';
import { z } from 'zod';

const client = new OpenAI({
  apiKey: 'dummy-key',
  baseURL: 'http://localhost:8000/v1',
});
setDefaultOpenAIClient(client);

// ========================================
// Define ALL cognition-cli tools
// ========================================

// Core file tools
const readFileTool = tool({
  name: 'read_file',
  description: 'Read file contents. Use offset/limit for large files.',
  parameters: z.object({
    file_path: z.string().describe('Absolute path to the file'),
    limit: z.number().nullable().describe('Max lines to read'),
    offset: z.number().nullable().describe('Line offset to start from'),
  }),
  execute: async ({ file_path, limit, offset }) => {
    console.log(`\n✅ [TOOL] read_file: ${file_path} (limit=${limit}, offset=${offset})`);
    return `File contents of ${file_path}:\nline 1: Hello World\nline 2: This is a test`;
  },
});

const globTool = tool({
  name: 'glob',
  description: 'Find files matching a glob pattern like "**/*.ts" or "src/*.py"',
  parameters: z.object({
    pattern: z.string().describe('Glob pattern'),
  }),
  execute: async ({ pattern }) => {
    console.log(`\n✅ [TOOL] glob: ${pattern}`);
    return `Found files:\nsrc/server.py\nsrc/chat.py\ntests/test_chat.py`;
  },
});

const grepTool = tool({
  name: 'grep',
  description: 'Search for pattern in files using ripgrep',
  parameters: z.object({
    pattern: z.string().describe('Regex pattern to search'),
  }),
  execute: async ({ pattern }) => {
    console.log(`\n✅ [TOOL] grep: ${pattern}`);
    return `src/chat.py:42: def ${pattern}():\nsrc/server.py:15: # ${pattern}`;
  },
});

// Mutating tools
const bashTool = tool({
  name: 'bash',
  description: 'Execute bash command (git, npm, ls, etc.)',
  parameters: z.object({
    command: z.string().describe('The command to execute'),
  }),
  execute: async ({ command }) => {
    console.log(`\n✅ [TOOL] bash: ${command}`);
    return `Exit code: 0\n$ ${command}\nCommand output here`;
  },
});

const writeFileTool = tool({
  name: 'write_file',
  description: 'Write content to a file at the given path',
  parameters: z.object({
    file_path: z.string().describe('Absolute path to write to'),
    content: z.string().describe('Content to write'),
  }),
  execute: async ({ file_path, content }) => {
    console.log(`\n✅ [TOOL] write_file: ${file_path} (${content.length} bytes)`);
    return `Successfully wrote ${content.length} bytes to ${file_path}`;
  },
});

const editFileTool = tool({
  name: 'edit_file',
  description: 'Replace text in a file (old_string must be unique)',
  parameters: z.object({
    file_path: z.string().describe('Absolute path to the file'),
    old_string: z.string().describe('Text to replace'),
    new_string: z.string().describe('Replacement text'),
  }),
  execute: async ({ file_path, old_string, new_string }) => {
    console.log(`\n✅ [TOOL] edit_file: ${file_path} ("${old_string}" -> "${new_string}")`);
    return `Successfully edited ${file_path}`;
  },
});

// Web tools
const fetchUrlTool = tool({
  name: 'fetch_url',
  description: 'Fetch content from a URL to read documentation, APIs, or external resources',
  parameters: z.object({
    url: z.string().describe('The URL to fetch content from'),
  }),
  execute: async ({ url }) => {
    console.log(`\n✅ [TOOL] fetch_url: ${url}`);
    return `Fetched content from ${url}:\n<title>Example Page</title>\nThis is the page content...`;
  },
});

const webSearchTool = tool({
  name: 'WebSearch',
  description: 'Search the web for information, current events, documentation',
  parameters: z.object({
    query: z.string().describe('The search query'),
  }),
  execute: async ({ query }) => {
    console.log(`\n✅ [TOOL] WebSearch: ${query}`);
    return `Search results for "${query}":\n1. Result One - https://example.com/1\n2. Result Two - https://example.com/2`;
  },
});

// Memory tools
const recallTool = tool({
  name: 'recall_past_conversation',
  description: 'Retrieve context from conversation history using semantic search',
  parameters: z.object({
    query: z.string().describe('What to search for in past conversation'),
  }),
  execute: async ({ query }) => {
    console.log(`\n✅ [TOOL] recall_past_conversation: ${query}`);
    return `Found relevant context:\nUser discussed "${query}" earlier in the conversation...`;
  },
});

// Background tasks
const getBackgroundTasksTool = tool({
  name: 'get_background_tasks',
  description: 'Query status of background operations (genesis, overlay generation)',
  parameters: z.object({
    filter: z.enum(['all', 'active', 'completed', 'failed']).nullable().default('all'),
  }),
  execute: async ({ filter }) => {
    console.log(`\n✅ [TOOL] get_background_tasks: filter=${filter || 'all'}`);
    return `**No active tasks** - all background operations idle.\n\n**Summary**: 3 total tasks\n  - Active: 0\n  - Completed: 3`;
  },
});

// Agent messaging/IPC tools
const listAgentsTool = tool({
  name: 'list_agents',
  description: 'List all active agents in the IPC bus. Shows who is online.',
  parameters: z.object({
    filter: z.string().optional().describe('Optional filter for agent status'),
  }),
  execute: async ({ filter }) => {
    console.log(`\n✅ [TOOL] list_agents${filter ? ` (filter=${filter})` : ''}`);
    return `**Active Agents (3):**\n\n| Alias | Model | Status |\n|-------|-------|--------|\n| opus1 | claude-opus | active |\n| sonnet2 | claude-sonnet | active |\n| gemma1 | gpt-oss-20b | active |`;
  },
});

const getPendingMessagesTool = tool({
  name: 'get_pending_messages',
  description: 'List pending messages in your message queue from other agents',
  parameters: z.object({
    limit: z.number().optional().describe('Max messages to return'),
  }),
  execute: async ({ limit }) => {
    console.log(`\n✅ [TOOL] get_pending_messages${limit ? ` (limit=${limit})` : ''}`);
    return `**Pending Messages (1):**\n\nFrom: opus1\nMessage: "Can you help with the authentication module?"`;
  },
});

const sendAgentMessageTool = tool({
  name: 'send_agent_message',
  description: 'Send a message to another agent by alias or ID',
  parameters: z.object({
    to: z.string().describe('Target agent alias or full agent ID'),
    message: z.string().describe('The message content to send'),
  }),
  execute: async ({ to, message }) => {
    console.log(`\n✅ [TOOL] send_agent_message: to=${to}, message="${message}"`);
    return `Message sent to ${to}: "${message}"`;
  },
});

const broadcastAgentMessageTool = tool({
  name: 'broadcast_agent_message',
  description: 'Broadcast a message to ALL active agents',
  parameters: z.object({
    message: z.string().describe('The message content to broadcast'),
  }),
  execute: async ({ message }) => {
    console.log(`\n✅ [TOOL] broadcast_agent_message: "${message}"`);
    return `Broadcast sent to 3 agents: "${message}"`;
  },
});

const markMessageReadTool = tool({
  name: 'mark_message_read',
  description: 'Mark a pending message as read/processed',
  parameters: z.object({
    messageId: z.string().describe('The message ID to mark as read'),
    status: z.enum(['read', 'injected', 'dismissed']).default('injected'),
  }),
  execute: async ({ messageId, status }) => {
    console.log(`\n✅ [TOOL] mark_message_read: ${messageId} -> ${status || 'injected'}`);
    return `Message ${messageId} marked as ${status || 'injected'}`;
  },
});

// All tools array
const allTools = [
  readFileTool, globTool, grepTool,
  bashTool, writeFileTool, editFileTool,
  fetchUrlTool, webSearchTool,
  recallTool, getBackgroundTasksTool,
  listAgentsTool, getPendingMessagesTool, sendAgentMessageTool,
  broadcastAgentMessageTool, markMessageReadTool,
];

// Create agent with all tools
const agent = new Agent({
  name: 'cognition_agent',
  model: 'gpt-oss-20b',
  instructions: `You are a helpful AI assistant with access to various tools.
Use tools proactively when needed. Match user intent to the right tool:
- "list files", "find files" → glob
- "read file", "show contents" → read_file
- "search for", "find pattern" → grep
- "run command", "execute" → bash
- "fetch URL", "get webpage" → fetch_url
- "search web", "google" → WebSearch
- "who is online", "list agents", "see agents" → list_agents
- "check messages", "pending messages" → get_pending_messages
- "send message to" → send_agent_message
- "broadcast", "tell everyone" → broadcast_agent_message
- "background tasks", "what's running" → get_background_tasks
- "recall", "what did we discuss" → recall_past_conversation`,
  tools: allTools,
});

// ========================================
// Test cases with natural language
// ========================================

const testCases = [
  // Agent messaging (user's specific request)
  { query: "who is online", expectedTool: "list_agents" },
  { query: "list agents", expectedTool: "list_agents" },
  { query: "check my messages", expectedTool: "get_pending_messages" },
  { query: "send hello to opus1", expectedTool: "send_agent_message" },

  // File operations
  { query: "list all python files", expectedTool: "glob" },
  { query: "read the server.py file", expectedTool: "read_file" },
  { query: "search for 'def main' in the codebase", expectedTool: "grep" },

  // Web operations
  { query: "fetch https://example.com", expectedTool: "fetch_url" },
  { query: "search the web for OpenAI agents SDK", expectedTool: "WebSearch" },

  // Background/memory
  { query: "what background tasks are running", expectedTool: "get_background_tasks" },
  { query: "what did we discuss about authentication", expectedTool: "recall_past_conversation" },
];

async function runTest(query, expectedTool) {
  console.log(`\n${'='.repeat(60)}`);
  console.log(`Query: "${query}"`);
  console.log(`Expected: ${expectedTool}`);
  console.log('-'.repeat(60));

  try {
    const result = await run(agent, query, { stream: true });

    // Consume the stream and collect ALL tool calls (not just last)
    const toolsCalled = [];
    for await (const event of result) {
      // Check run_item_stream_event for tool calls
      if (event.type === 'run_item_stream_event') {
        const item = event.item;
        // SDK uses 'tool_call_item' type for tool calls
        if (item?.type === 'tool_call_item' || item?.type === 'function_call') {
          const name = item.rawItem?.name || item.name;
          if (name && !toolsCalled.includes(name)) {
            toolsCalled.push(name);
          }
        }
      }
    }

    // Also check newItems after stream completes (more reliable)
    if (toolsCalled.length === 0 && result.newItems) {
      for (const item of result.newItems) {
        // SDK uses 'tool_call_item' type
        if (item.type === 'tool_call_item' || item.type === 'function_call') {
          const name = item.rawItem?.name || item.name;
          if (name && !toolsCalled.includes(name)) {
            toolsCalled.push(name);
          }
        }
      }
    }

    // Check if expected tool was called (among all tools used)
    const success = toolsCalled.includes(expectedTool);
    const toolCalled = toolsCalled[0] || null; // First tool for display
    console.log(`\nResult: ${success ? '✅ PASS' : '❌ FAIL'}`);
    console.log(`Tools called: ${toolsCalled.length > 0 ? toolsCalled.join(', ') : 'none'}`);

    if (!success) {
      // Show what we got for debugging
      const finalText = result.finalOutput || '';
      console.log(`Final output: ${finalText.substring(0, 200)}${finalText.length > 200 ? '...' : ''}`);
      if (result.newItems?.length) {
        console.log(`Items: ${result.newItems.map(i => `${i.type}:${i.rawItem?.name || ''}`).join(', ')}`);
      }
    }

    return success;
  } catch (error) {
    console.log(`\n❌ ERROR: ${error.message}`);
    if (error.cause) console.log(`   Cause: ${error.cause}`);
    return false;
  }
}

// Run all tests
console.log('Testing ALL cognition-cli tools with natural language\n');
console.log(`Endpoint: ${client.baseURL}`);
console.log(`Total tests: ${testCases.length}`);

let passed = 0;
let failed = 0;

for (const { query, expectedTool } of testCases) {
  const success = await runTest(query, expectedTool);
  if (success) passed++;
  else failed++;

  // Small delay between tests to avoid rate limiting
  await new Promise(r => setTimeout(r, 1000));
}

console.log(`\n${'='.repeat(60)}`);
console.log(`SUMMARY: ${passed}/${testCases.length} passed, ${failed} failed`);
console.log('='.repeat(60));
