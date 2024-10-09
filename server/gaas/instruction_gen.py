import os
import logging
from typing import Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from globals.globals import set_server_status

logger = logging.getLogger(__name__)


class InstructionGen:
    def __init__(
        self,
        model_name: str = "microsoft/Phi-3-mini-128k-instruct",
        device: str = "cuda:0",
        model_files_local: Optional[bool] = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if model_files_local:
            model_files_path = os.path.join(script_dir, "..", "..", "model_files")
        else:
            model_files_path = "/app/model_files/phi-3-mini-128k-instruct"

        self.model_name = model_name
        self.seed = torch.random.manual_seed(0)

        try:
            logger.info("Loading the model from path: %s", model_files_path)
            logger.info("Initializing InstructionGen...")
            set_server_status("Initializing InstructionGen...")

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="cuda",
                torch_dtype="auto",
                trust_remote_code=True,
            )

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=tokenizer,
            )

        except Exception as e:
            logger.error("Failed to initialize InstructionGen: %s", e)
            set_server_status("InstructionGen FAILED to initialize.")
            raise

    def ask_model(
        self, prompt: str, temp: float = 0.1, prob: float = 0.2, max_tokens: int = 1024
    ):
        """
        Args:
            prompt (_type_): prompt to execute
            temp (float, optional): temperature for the model. Defaults to 0.1.
            prob (float, optional): nucleous sampling, controls randomness of the output. Defaults to 0.2.
            max_tokens (int, optional): max number of tokens in the output. Defaults to 1024.

        Returns:
            string: response message
        """
        generation_args = {
            "max_new_tokens": max_tokens,
            "return_full_text": False,
            "temperature": temp,
            "top_p": prob,
            "do_sample": True,
        }
        messages = [
            {"role": "user", "content": prompt},
        ]
        output = self.pipe(messages, **generation_args)
        out = output[0]["generated_text"]
        return out

    def detect_task_target(
        self, task: str, temp: float = 0.1, prob: float = 0.2, max_tokens: float = 256
    ):
        """
        Args:
            task (str): the definition of the task
            temp (float, optional): temperature for the model. Defaults to 0.1.
            prob (float, optional): nucleous sampling, controls randomness of the output. Defaults to 0.2.
            max_tokens (float, optional): max number of tokens in the output. Defaults to 256.

        Returns:
            _type_: The target for the task
        """
        system = f"""Imagine you have a task {task}. 
                Explain in a single sentence who the task should be intended for. 
                """
        prompt = f"### System:\n{system}\n### Response:\n"
        text = self.ask_model(prompt, temp, prob, max_tokens)
        out = text.split("</s>")[0]
        return out

    def generate(
        self,
        task: str,
        temp: float = 0.1,
        prob: float = 0.2,
        max_tokens: float = 1024,
        **kwargs,
    ):
        """
        Args:
            task (str): task to decompose
            temp (float, optional): temperature for the model. Defaults to 0.1.
            prob (float, optional): nucleous sampling, controls randomness of the output. Defaults to 0.2.
            max_tokens (float, optional): max number of tokens in the outputa. Defaults to 1024.

        Returns:
            _type_: Sequence of sub-tasks
        """
        task_target = self.detect_task_target(task, temp, prob, max_tokens)
        system = (
            task_target
            + """"
            ###Context:
            When faced with a large, complex task we need to employ a systematic approach to break it down into more manageable sub-tasks. This process involves analyzing the task, 
            understanding its components for each sub-task. Throughout the decomposition process, iteratively define the methodology for each sub-task, which includes concrete instructions 
            or algorithms specific to that sub-task. Each step or instruction within the methodology should be atomic, clear, and actionable. Based on the defined methodology evaluate the size 
            and complexity of each sub-task and further divide it into smaller steps if necessary. This process is recursive until all subtasks, inputs, outputs, constraints, and Methodologies 
            are thoroughly defined. Then proceed with task execution utilizing the output from previous steps as input for subsequent tasks.

            ###Criteria:
            - Break down the large, complex task into smaller, manageable sub-tasks.
            - Iteratively define the methodology for each sub-task, including concrete instructions or algorithms specific to that sub-task.
            - Ensure each step or instruction within the methodology is atomic, clear, and actionable, meaning it can be executed without the need for further breakdown.
            - Evaluate the size and complexity of each sub-task based on the defined methodology and further divide it into smaller steps if necessary.
            - Ensure all sub-tasks, inputs, outputs, constraints, and methodologies are thoroughly defined before proceeding with task execution.
            - Present the complete sub-task structure, including well-defined input, output, methodology, and possibly constraints for each sub-task.
            - Utilize the output from completed sub-tasks as input for subsequent tasks.
            - Ensure the successful completion of the entire task by effectively managing the task decomposition.

            Use the format: {{"Steps": [{{"Title": "title", "Action": "detailed action to perform", "Input Data": "input data items", "Output Data": "output data items",
            "Methodology":"defined methodology for the action","Assessment": "action assessment"}}]}}
            """
        )
        prompt = f"### System:\n{system}\n### Task:\n{task}\n### Response:"
        out = self.ask_model(prompt, temp, prob, max_tokens)
        start = out.find("{")
        end = out.rfind("}")
        out = out[start : (end + 1)]
        return {"data": out}
