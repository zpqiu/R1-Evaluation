SKY_T1_SYS = "Your role as an assistant involves thoroughly exploring questions through a systematic long \
        thinking process before providing the final precise and accurate solutions. This requires \
        engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, \
        backtracing, and iteration to develop well-considered thinking process. \
        Please structure your response into two main sections: Thought and Solution. \
        In the Thought section, detail your reasoning process using the specified format: \
        <|begin_of_thought|> {thought with steps separated with '\n\n'} \
        <|end_of_thought|> \
        Each step should include detailed considerations such as analisying questions, summarizing \
        relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining \
        any errors, and revisiting previous steps. \
        In the Solution section, based on various attempts, explorations, and reflections from the Thought \
        section, systematically present the final solution that you deem correct. The solution should \
        remain a logical, accurate, concise expression style and detail necessary step needed to reach the \
        conclusion, formatted as follows: \
        <|begin_of_solution|> \
        {final formatted, precise, and clear solution} \
        <|end_of_solution|> \
        Now, try to solve the following question through the above guidelines:"

R1_SYS = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process is enclosed within <think> </think>, i.e., "
    "<think> reasoning process here </think>."
)

SYSTEM_PROMPT = {
    "Qwen/Qwen2-7B-Instruct": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step.",
    "Qwen/QwQ-32B-Preview": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step.",
    "Qwen/Qwen2.5-72B-Instruct": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step.",
    "Qwen/Qwen2.5-32B-Instruct": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step.",
    "Qwen/Qwen2.5-7B-Instruct": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step.",
    "Qwen/Qwen2.5-1.5B-Instruct": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step.",
    "Qwen/Qwen2.5-Math-7B-Instruct": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step.",
    "meta-llama/Llama-3.2-1B-Instruct":  "You are a helpful and harmless assistant. You are Llama developed by Meta. You should think step-by-step.",
    "bespokelabs/Bespoke-Stratos-7B": SKY_T1_SYS,
    "R1/Qwen2.5-1.5B-Instruct": SKY_T1_SYS,
    "./R1-3B-3096": R1_SYS,
    "./global_step25_hf": "",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": "",
}

MODEL_TO_NAME = {
    "Qwen/Qwen2-7B-Instruct": "Qwen2-7B-Instruct",
    "Qwen/QwQ-32B-Preview": "QwQ-32B-Preview",
    "Qwen/Qwen2.5-72B-Instruct": "Qwen2.5-72B-Instruct",
    "Qwen/Qwen2.5-32B-Instruct": "Qwen2.5-32B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct": "Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct": "Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-Math-7B-Instruct": "Qwen2.5-Math-7B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct": "Llama-3.2-1B-Instruct",
    "deepseek-reasoner": "R1",
    "bespokelabs/Bespoke-Stratos-7B": "Bespoke-Stratos-7B",
    "R1/Qwen2.5-1.5B-Instruct": "R1-Qwen2.5-1.5B-Instruct",
    "R1/Qwen2.5-7B-RLOO": "R1-Qwen2.5-7B-RLOO",
    "./R1-3B-3096": "R1-3B-3096",
    "./global_step25_hf": "global_step25_hf",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": "DeepSeek-R1-Distill-Qwen-7B"
}
