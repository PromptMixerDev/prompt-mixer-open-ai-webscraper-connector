import OpenAI from 'openai';
import { config } from './config.js';
import { ChatCompletion } from 'openai/resources';
import fetch from 'node-fetch';
import * as cheerio from 'cheerio';

const API_KEY = 'API_KEY';

interface Message {
  role: string;
  content: string;
  tool_call_id?: string | null;
  name?: string | null;
}

interface Completion {
  Content: string | null;
  Error?: string | undefined;
  TokenUsage: number | undefined;
  ToolCalls?: any;
}

interface ConnectorResponse {
  Completions: Completion[];
  ModelType: string;
}

interface ErrorCompletion {
  choices: Array<{
    message: {
      content: string;
    };
  }>;
  error: string;
  model: string;
  usage: undefined;
}

type WebpageFunction = (...args: any[]) => Promise<any>;

interface AvailableFunctions {
  [key: string]: WebpageFunction;
}

const mapToResponse = (
  outputs: Array<ChatCompletion | ErrorCompletion>,
  model: string,
): ConnectorResponse => {
  return {
    Completions: outputs.map((output) => {
      if ('error' in output) {
        return {
          Content: null,
          TokenUsage: undefined,
          Error: output.error,
        };
      } else {
        return {
          Content: output.choices[0]?.message?.content,
          TokenUsage: output.usage?.total_tokens,
        };
      }
    }),
    ModelType: outputs[0].model || model,
  };
};

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const mapErrorToCompletion = (error: any, model: string): ErrorCompletion => {
  const errorMessage = error.message || JSON.stringify(error);
  return {
    choices: [],
    error: errorMessage,
    model,
    usage: undefined,
  };
};

// Type guard to check if a node is an Element
function isElement(node: cheerio.Element): node is cheerio.Element & { tagName: string } {
  return node.type === 'tag' && 'tagName' in node;
}

function parseNodeToMarkdown(node: cheerio.Cheerio, $: cheerio.Root): string {
  let markdown = '';

  switch (node[0].type) {
    case 'tag':
      const element = node;
      if (isElement(element[0])) {
        const tagName = element[0].tagName.toLowerCase();
        switch (tagName) {
          case 'h1':
            markdown += `# ${element.text()}\n\n`;
            break;
          case 'h2':
            markdown += `## ${element.text()}\n\n`;
            break;
          case 'h3':
            markdown += `### ${element.text()}\n\n`;
            break;
          case 'h4':
            markdown += `#### ${element.text()}\n\n`;
            break;
          case 'h5':
            markdown += `##### ${element.text()}\n\n`;
            break;
          case 'h6':
            markdown += `###### ${element.text()}\n\n`;
            break;
          case 'p':
            markdown += `${element.text()}\n\n`;
            break;
          case 'a':
            markdown += `[${element.text()}](${element.attr('href')})`;
            break;
          case 'img':
            markdown += `![${element.attr('alt')}](${element.attr('src')})`;
            break;
          case 'ul':
            element.children('li').each((index: number, child: cheerio.Element) => {
              markdown += `- ${$(child).text()}\n`;
            });
            markdown += '\n';
            break;
          case 'ol':
            let counter = 1;
            element.children('li').each((index: number, child: cheerio.Element) => {
              markdown += `${counter}. ${$(child).text()}\n`;
              counter++;
            });
            markdown += '\n';
            break;
          default:
            element.contents().each((index: number, childNode: cheerio.Element) => {
              markdown += parseNodeToMarkdown($(childNode), $);
            });
        }
      }
      break;

    case 'text':
      markdown += node.text();
      break;
  }

  return markdown;
}

async function parseWebpageToMarkdown(url: string) {
  const response = await fetch(url);
  console.log('Response:', response);
  if (!response.ok) {
    throw new Error('Failed to fetch the webpage: ' + response.statusText);
  }
  const html = await response.text();
  const $ = cheerio.load(html);
  const root = $('body');
  let markdown = '';

  root.contents().each((index: number, childNode: cheerio.Element) => {
    markdown += parseNodeToMarkdown($(childNode), $);
  });

  return markdown;
}

async function main(
  model: string,
  prompts: string[],
  properties: Record<string, unknown>,
  settings: Record<string, unknown>,
) {
  const openai = new OpenAI({
    apiKey: settings?.[API_KEY] as string,
  });

  const total = prompts.length;
  const { prompt, ...restProperties } = properties;
  const systemPrompt = (prompt ||
    config.properties.find((prop) => prop.id === 'prompt')?.value) as string;
  const messageHistory: Message[] = [{ role: 'system', content: systemPrompt }];
  const outputs: Array<ChatCompletion | ErrorCompletion> = [];

  const tools = [
    {
      "type": "function",
      "function": {
        "name": "parseWebpageToMarkdown",
        "description": "Parse a webpage into Markdown format",
        "parameters": {
          "type": "object",
          "properties": {
            "url": {
              "type": "string",
              "description": "The URL of the webpage to parse"
            }
          },
          "required": ["url"]
        }
      }
    }
  ];

  try {
    for (let index = 0; index < total; index++) {
      try {
        messageHistory.push({ role: 'user', content: prompts[index] });
        const chatCompletion = await openai.chat.completions.create({
          messages: messageHistory as unknown as [],
          model,
          tools: tools.map(tool => ({ type: "function", function: tool.function })),
          tool_choice: "auto",
          ...restProperties,
        });

        const assistantResponse = chatCompletion.choices[0].message.content || 'No response.';
        messageHistory.push({ role: 'assistant', content: assistantResponse });

        console.log('Chat completion:', chatCompletion);

        // Check if the assistant's response contains a tool call
        const toolCalls = chatCompletion.choices[0].message.tool_calls;
        if (toolCalls) {
          const availableFunctions: AvailableFunctions = {
            parseWebpageToMarkdown: parseWebpageToMarkdown
          };
          for (const toolCall of toolCalls) {
            const functionName = toolCall.function.name;
            const functionToCall = availableFunctions[functionName];
            const functionArgs = JSON.parse(toolCall.function.arguments);
            console.log('Function arguments:', functionArgs);
            const functionResponse = await functionToCall(
              functionArgs.url
            );
            messageHistory.push({
              tool_call_id: toolCall.id,
              role: "function",
              name: functionName,
              content: functionResponse,
            });
          }
          const secondResponse = await openai.chat.completions.create({
            model: model,
            messages: messageHistory as unknown as [],
            ...restProperties,
          });
          const secondAssistantResponse = secondResponse.choices[0].message.content || 'No response.';
          outputs.push(secondResponse);
          messageHistory.push({ role: 'assistant', content: secondAssistantResponse });
        } else {
          outputs.push(chatCompletion);
        }

      } catch (error) {
        console.error('Error in main loop:', error);
        const completionWithError = mapErrorToCompletion(error, model);
        outputs.push(completionWithError);
      }
    }

    return mapToResponse(outputs, model);
  } catch (error) {
    console.error('Error in main function:', error);
    return { Error: error, ModelType: model };
  }
}

export { main, config };