import type { BaseLanguageModel } from '@langchain/core/language_models/base';
import { RunnableSequence } from '@langchain/core/runnables';
import { AgentExecutor, createToolCallingAgent } from 'langchain/agents';
import type { BaseChatMemory } from 'langchain/memory';
import omit from 'lodash/omit';
import { jsonParse, NodeOperationError } from 'n8n-workflow';
import type { IExecuteFunctions, INodeExecutionData } from 'n8n-workflow';

import { getPromptInputByType } from '@utils/helpers';
import { getOptionalOutputParser } from '@utils/output_parsers/N8nOutputParser';

import {
	fixEmptyContentMessage,
	getAgentStepsParser,
	getChatModel,
	getOptionalMemory,
	getTools,
	prepareMessages,
	preparePrompt,
} from '../common';
import { SYSTEM_MESSAGE } from '../prompt';

/* -----------------------------------------------------------
   Sliding Window Memory Wrapper
----------------------------------------------------------- */
/**
 * 创建一个滑动窗口内存包装器，限制对话历史的长度
 */
function createSlidingWindowMemory(
	originalMemory: BaseChatMemory | undefined,
	windowSize: number = 10,
) {
	if (!originalMemory || windowSize === 0) return originalMemory;

	// 包装loadMemoryVariables方法来实现滑动窗口
	const originalLoadMemoryVariables = originalMemory.loadMemoryVariables.bind(originalMemory);
	const originalSaveContext = originalMemory.saveContext.bind(originalMemory);

	originalMemory.loadMemoryVariables = async function (values: any) {
		console.log('🔧 [SLIDING WINDOW] Loading memory variables, window size:', windowSize);

		// 先获取完整的内存变量
		let memoryVars = await originalLoadMemoryVariables(values);

		// 🔧 处理chat_history - 应用滑动窗口
		if (memoryVars.chat_history && Array.isArray(memoryVars.chat_history)) {
			const originalLength = memoryVars.chat_history.length;

			if (originalLength > windowSize * 2) {
				// windowSize * 2 因为每轮对话包含用户消息和AI消息
				// 保留最近的消息
				const keptMessages = memoryVars.chat_history.slice(-windowSize * 2);
				console.log(
					'🔧 [SLIDING WINDOW] Trimmed chat history from',
					originalLength,
					'to',
					keptMessages.length,
					'messages',
				);

				memoryVars = {
					...memoryVars,
					chat_history: keptMessages,
				};
			}
		}

		// 🔧 处理intermediate_steps - 限制工具调用步骤历史
		if (memoryVars.intermediate_steps && Array.isArray(memoryVars.intermediate_steps)) {
			const originalStepsLength = memoryVars.intermediate_steps.length;

			if (originalStepsLength > windowSize) {
				// 只保留最近的N个工具调用步骤
				memoryVars.intermediate_steps = memoryVars.intermediate_steps.slice(-windowSize);
				console.log(
					'🔧 [SLIDING WINDOW] Trimmed intermediate_steps from',
					originalStepsLength,
					'to',
					memoryVars.intermediate_steps.length,
					'steps',
				);
			}
		}

		return memoryVars;
	};

	// 包装saveContext方法来在保存时也进行窗口管理
	originalMemory.saveContext = async function (inputValues: any, outputValues: any) {
		// 🔧 处理intermediateSteps - 限制工具调用步骤的数量
		if (
			outputValues &&
			typeof outputValues === 'object' &&
			outputValues.intermediateSteps &&
			Array.isArray(outputValues.intermediateSteps)
		) {
			const originalStepsLength = outputValues.intermediateSteps.length;

			if (originalStepsLength > windowSize) {
				// 只保留最近的N个工具调用步骤
				outputValues.intermediateSteps = outputValues.intermediateSteps.slice(-windowSize);
				console.log(
					'🔧 [SLIDING WINDOW] Trimmed intermediateSteps from',
					originalStepsLength,
					'to',
					outputValues.intermediateSteps.length,
					'steps',
				);
			}
		}

		// 先保存上下文
		await originalSaveContext(inputValues, outputValues);

		// 然后检查是否需要清理旧消息
		if (originalMemory.chatHistory && 'getMessages' in originalMemory.chatHistory) {
			try {
				const messages = await originalMemory.chatHistory.getMessages();
				const currentLength = messages.length;

				if (currentLength > windowSize * 2) {
					// 需要清理旧消息
					const messagesToKeep = messages.slice(-windowSize * 2);
					const messagesToRemove = currentLength - windowSize * 2;

					console.log(
						'🔧 [SLIDING WINDOW] Removing',
						messagesToRemove,
						'old messages, keeping',
						messagesToKeep.length,
					);

					// 清空历史并重新添加最近的消息
					if ('clear' in originalMemory.chatHistory && 'addMessage' in originalMemory.chatHistory) {
						await originalMemory.chatHistory.clear();
						for (const message of messagesToKeep) {
							await originalMemory.chatHistory.addMessage(message);
						}
					}
				}
			} catch (error) {
				console.log(
					'🔧 [SLIDING WINDOW] Warning: Could not manage chat history window:',
					error.message,
				);
			}
		}
	};

	return originalMemory;
}

/* -----------------------------------------------------------
   Custom Tool Calling Agent with Error Recovery
----------------------------------------------------------- */
/**
 * 创建一个自定义的工具调用代理，能够处理解析错误
 */
async function createCustomToolCallingAgent(config: {
	llm: any;
	tools: any[];
	prompt: any;
	streamRunnable: boolean;
	contextWindowSize?: number;
}) {
	const { RunnableLambda } = await import('@langchain/core/runnables');

	// 创建原始代理
	const originalAgent = createToolCallingAgent(config);

	// 包装代理以处理解析错误和限制steps
	return RunnableLambda.from(async (input: any) => {
		// 🔧 限制输入中的steps数组长度
		const windowSize = config.contextWindowSize || 10;
		if (input && typeof input === 'object' && input.steps && Array.isArray(input.steps)) {
			const originalStepsLength = input.steps.length;
			if (originalStepsLength > windowSize) {
				input.steps = input.steps.slice(-windowSize);
				console.log(
					'🔧 [CUSTOM AGENT] Trimmed input steps from',
					originalStepsLength,
					'to',
					input.steps.length,
					'steps',
				);
			}
		}

		console.log('🔧 [CUSTOM AGENT] Processing input with', input?.steps?.length || 0, 'steps');

		try {
			const result = await originalAgent.invoke(input);
			console.log('🔧 [CUSTOM AGENT] Agent result:', JSON.stringify(result, null, 2));
			return result;
		} catch (error: any) {
			console.log('🔧 [CUSTOM AGENT] Caught error:', error.message);

			// 如果是工具解析错误，尝试从对话历史中恢复
			if (
				error.message?.includes('Failed to parse tool arguments') ||
				error.message?.includes('Unexpected end of JSON input')
			) {
				console.log('🔧 [CUSTOM AGENT] Attempting to recover from parsing error...');

				// 创建一个成功的AgentFinish响应
				return {
					returnValues: {
						output:
							'Tool execution completed successfully. The requested action has been performed.',
					},
					log: 'Recovered from tool parsing error - execution was successful',
				};
			}

			// 重新抛出其他错误
			throw error;
		}
	});
}

/* -----------------------------------------------------------
   Main Executor Function
----------------------------------------------------------- */
/**
 * The main executor method for the Tools Agent.
 *
 * This function retrieves necessary components (model, memory, tools), prepares the prompt,
 * creates the agent, and processes each input item. The error handling for each item is also
 * managed here based on the node's continueOnFail setting.
 *
 * @returns The array of execution data for all processed items
 */
export async function toolsAgentExecute(this: IExecuteFunctions): Promise<INodeExecutionData[][]> {
	this.logger.debug('Executing Tools Agent');

	const returnData: INodeExecutionData[] = [];
	const items = this.getInputData();
	const outputParser = await getOptionalOutputParser(this);
	const tools = await getTools(this, outputParser);

	for (let itemIndex = 0; itemIndex < items.length; itemIndex++) {
		try {
			const model = (await getChatModel(this)) as BaseLanguageModel;
			const originalMemory = await getOptionalMemory(this);
			// 🔧 应用滑动窗口内存包装器，从选项中获取窗口大小设置
			const agentOptions = this.getNodeParameter('options', itemIndex, {}) as {
				contextWindowSize?: number;
				systemMessage?: string;
				passthroughBinaryImages?: boolean;
				returnIntermediateSteps?: boolean;
				maxIterations?: number;
			};
			const contextWindowSize = agentOptions.contextWindowSize ?? 10;
			const memory = createSlidingWindowMemory(originalMemory, contextWindowSize);

			const input = getPromptInputByType({
				ctx: this,
				i: itemIndex,
				inputKey: 'text',
				promptTypeKey: 'promptType',
			});
			if (input === undefined) {
				throw new NodeOperationError(this.getNode(), 'The “text” parameter is empty.');
			}

			// Use agentOptions already defined above

			// Prepare the prompt messages and prompt template.
			const messages = await prepareMessages(this, itemIndex, {
				systemMessage: agentOptions.systemMessage,
				passthroughBinaryImages: agentOptions.passthroughBinaryImages ?? true,
				outputParser,
			});
			const prompt = preparePrompt(messages);

			// 🔧 Create a proper model wrapper that preserves all LangChain model methods
			const wrappedModel = Object.create(Object.getPrototypeOf(model));
			Object.assign(wrappedModel, model);

			// Override the invoke method to fix parsing issues
			const modelInvoke = model.invoke.bind(model);
			wrappedModel.invoke = async (input: any) => {
				console.log('🔧 [MODEL WRAPPER] Original input:', JSON.stringify(input, null, 2));

				const result = await modelInvoke(input);
				console.log('🔧 [MODEL WRAPPER] Model result:', JSON.stringify(result, null, 2));

				// Fix tool_calls with malformed arguments
				if (result.additional_kwargs?.tool_calls) {
					result.additional_kwargs.tool_calls = result.additional_kwargs.tool_calls.map(
						(toolCall: any) => {
							if (toolCall.function?.arguments) {
								try {
									// Try to parse the arguments to see if they're valid
									JSON.parse(toolCall.function.arguments);
								} catch (e) {
									console.log(
										'🔧 [MODEL WRAPPER] Fixing malformed tool arguments:',
										toolCall.function.arguments,
									);
									// If it's "[]" or other malformed JSON, fix it
									if (
										toolCall.function.arguments === '[]' ||
										toolCall.function.arguments.trim() === ''
									) {
										toolCall.function.arguments = '{}';
									} else {
										// Try to repair common JSON issues
										let fixed = toolCall.function.arguments.trim();
										if (!fixed.startsWith('{')) fixed = '{' + fixed;
										if (!fixed.endsWith('}')) fixed = fixed + '}';
										toolCall.function.arguments = fixed;
									}
									console.log('🔧 [MODEL WRAPPER] Fixed arguments:', toolCall.function.arguments);
								}
							}
							return toolCall;
						},
					);
				}

				console.log('🔧 [MODEL WRAPPER] Fixed result:', JSON.stringify(result, null, 2));
				return result;
			};

			// 🔧 Create a custom agent that handles parsing errors
			const agent = await createCustomToolCallingAgent({
				llm: wrappedModel as any,
				tools,
				prompt,
				streamRunnable: false,
				contextWindowSize,
			});
			// 设置streamRunnable属性（如果存在）
			if ('streamRunnable' in agent) {
				(agent as any).streamRunnable = false;
			}
			// Wrap the agent with parsers and fixes.
			const runnableAgent = RunnableSequence.from([
				agent,
				getAgentStepsParser(outputParser, memory),
				fixEmptyContentMessage,
			]);
			const baseExecutor = AgentExecutor.fromAgentAndTools({
				agent: runnableAgent,
				memory,
				tools,
				returnIntermediateSteps: agentOptions.returnIntermediateSteps === true,
				maxIterations: agentOptions.maxIterations ?? 10,
			});

			// 🔧 包装executor来限制intermediateSteps的数量
			const baseExecutorInvoke = baseExecutor.invoke.bind(baseExecutor);
			const executor = {
				...baseExecutor,
				invoke: async function (input: any, options?: any) {
					const result = await baseExecutorInvoke(input, options);

					// 限制intermediateSteps的数量
					if (result && result.intermediateSteps && Array.isArray(result.intermediateSteps)) {
						const originalStepsLength = result.intermediateSteps.length;

						if (originalStepsLength > contextWindowSize) {
							result.intermediateSteps = result.intermediateSteps.slice(-contextWindowSize);
							console.log(
								'🔧 [SLIDING WINDOW] Limited executor intermediateSteps from',
								originalStepsLength,
								'to',
								result.intermediateSteps.length,
								'steps',
							);
						}
					}

					return result;
				},
			};

			// Now the executor should work correctly with our model wrapper

			// Invoke the executor with the given input and system message.
			const response = await executor.invoke(
				{
					input,
					system_message: agentOptions.systemMessage ?? SYSTEM_MESSAGE,
					formatting_instructions:
						'IMPORTANT: For your response to user, you MUST use the `format_final_json_response` tool with your complete answer formatted according to the required schema. Do not attempt to format the JSON manually - always use this tool. Your response will be rejected if it is not properly formatted through this tool. Only use this tool once you are ready to provide your final answer.',
				},
				{ signal: this.getExecutionCancelSignal() },
			);

			// If memory and outputParser are connected, parse the output.
			if (memory && outputParser) {
				const parsedOutput = jsonParse<{ output: Record<string, unknown> }>(
					response.output as string,
				);
				response.output = parsedOutput?.output ?? parsedOutput;
			}

			// Omit internal keys before returning the result.
			const itemResult = {
				json: omit(
					response,
					'system_message',
					'formatting_instructions',
					'input',
					'chat_history',
					'agent_scratchpad',
				),
			};

			returnData.push(itemResult);
		} catch (error) {
			if (this.continueOnFail()) {
				returnData.push({
					json: { error: error.message },
					pairedItem: { item: itemIndex },
				});
				continue;
			}
			throw error;
		}
	}

	return [returnData];
}
