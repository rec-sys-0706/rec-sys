import logging
import json
from openai import OpenAI
from openai.types.beta.thread import Thread
from openai.types.beta.threads.run import Run
from thread import retrieve_thread, retrieve_thread_messages, create_thread, create_message
from utils import save_jsonl
client = OpenAI()
class ChatAgent:
    def __init__(self, assistant_id):
        if assistant_id is not None:
            self.assistant = client.beta.assistants.retrieve(assistant_id)
    def run(self, thread: Thread) -> str:
        """"""
        run = client.beta.threads.runs.create_and_poll(assistant_id=self.assistant.id, thread_id=thread.id)
        # TODO
        # try:

        #         # run = client.beta.threads.create_and_run_poll(
        #         #     assistant_id=self.assistant.id,
        #         #     thread={
        #         #         'messages': [
        #         #             {
        #         #                 'role': 'user',
        #         #                 'content': prompt
        #         #             }
        #         #         ]
        #         #     },
        #         #     tool_choice={'type': 'function', 'function': {'name': 'insert_row'}}
        #         # )
        #         # retrieve_thread(run.thread_id, self.log_path)
        #         # # print(run.to_json())
        #         # response = self.run_handler(run)
        #         # print(response)
        # except Exception as e:
        #     print(e)
        # finally:
        #     self.df.to_csv(self.output_path, index=False)
    def run_handler(self, run: Run) -> str:
        logging.info(f'[Run] Handling run with ID: {run.id}')

        if run.status == 'completed':
            # Return the latest message in thread.
            logging.info(f'[Run] Run with ID: {run.id} is completed with usage:\n{run.usage.to_json()}')
            messages = retrieve_thread_messages(run.thread_id)
            return messages[0].content[0].text.value
        elif run.status == 'requires_action':
            tool_calls = run.required_action.submit_tool_outputs.tool_calls

            tool_outputs = []
            for tool_call in tool_calls:
                arguments = json.loads(tool_call.function.arguments)
                for tool in self.tools:
                    if tool_call.function.name == tool['function']['name']:
                        args = [arguments.get(param) for param in self.tools[0]['function']['parameters']['properties'].keys()]

                        method = getattr(self, tool_call.function.name, None)
                        if callable(method):
                            output = method(*args)
                        else:
                            raise ValueError(f"No such method: {tool_call.function.name}")

                        tool_outputs.append({
                            'tool_call_id': tool_call.id,
                            'output': output
                        })

            if tool_outputs:
                run = client.beta.threads.runs.submit_tool_outputs_and_poll(
                    thread_id=run.thread_id,
                    run_id=run.id,
                    tool_outputs=tool_outputs
                )
            else:
                print("No tool outputs to submit.") # TODO delete
            return self.run_handler(run)
        else:
            raise ValueError(run.status) # TODO 
