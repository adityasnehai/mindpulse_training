"""Shared prompt construction for MindPulse's LLM task: turn one SFT record's
`input` dict into the user turn, and `output` dict into the target model turn,
using Gemma 3's real chat template (verified: `<start_of_turn>user ...
<end_of_turn><start_of_turn>model ...<end_of_turn>`).
"""

import json

TASK_INSTRUCTION = (
    "You are MindPulse, an on-device assistant. Given structured evidence about "
    "a behavioural routine change, respond with ONLY a single JSON object with "
    'keys "acknowledgement" (neutral, non-diagnostic, <=200 chars), "question" '
    '(one gentle reflective question, <=200 chars), and "action" (an object with '
    '"id" chosen from candidate_actions, "title", "duration_minutes" matching '
    "available_time_minutes, and up to 3 \"steps\"). Do not diagnose, suggest "
    "medication, or claim certainty about someone's emotions. No text outside the JSON object."
)


def build_user_turn(llm_input: dict) -> str:
    return f"{TASK_INSTRUCTION}\n\nInput:\n{json.dumps(llm_input)}"


def build_model_turn(llm_output: dict) -> str:
    return json.dumps(llm_output)


def build_messages(record: dict) -> list[dict]:
    # role must be "assistant" (the transformers chat-template convention) —
    # Gemma's template translates that internally into the literal
    # "<start_of_turn>model" marker. Passing the literal string "model" here
    # is not recognized by the template logic and silently breaks the
    # prompt/target token boundary (caught by a real-tokenizer test: with
    # role="model" the entire sequence came back masked).
    return [
        {"role": "user", "content": build_user_turn(record["input"])},
        {"role": "assistant", "content": build_model_turn(record["output"])},
    ]


def tokenize_and_mask(record: dict, tokenizer, max_length: int) -> dict:
    """Tokenizes the full user+model turn pair and masks every token up to
    and including the "<start_of_turn>model\\n" marker with label=-100, so
    the loss only trains on the JSON output, never the prompt — standard SFT
    practice, not something the base chat-template API does for you.

    Uses tokenize=False (get the formatted string) then tokenizes explicitly,
    rather than tokenize=True — verified against the real tokenizer that
    apply_chat_template(..., tokenize=True) returns a BatchEncoding here, and
    len(BatchEncoding) is the number of FIELDS (2: input_ids/attention_mask),
    not the token count. Using tokenize=True directly silently produced a
    prompt_len of 2 regardless of actual content, masking either everything
    or nothing depending on sequence length — caught by a real-tokenizer
    test asserting the JSON output text survives unmasked, not assumed correct
    just because no exception was raised.
    """
    messages = build_messages(record)
    prompt_only = messages[:1]

    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prompt_text = tokenizer.apply_chat_template(prompt_only, tokenize=False, add_generation_prompt=True)

    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

    prompt_len = len(prompt_ids)
    full_ids = full_ids[:max_length]
    labels = list(full_ids)
    for i in range(min(prompt_len, len(labels))):
        labels[i] = -100

    return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}
