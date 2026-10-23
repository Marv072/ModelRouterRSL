"""Output contract adapter; never sees or inserts the reference answer."""
import json


def structured_prompt(case):
    task = case['prompt'].replace('Return only the integer, no explanation.', '')
    task = task.replace('Return only a JSON array of their id strings, no markdown.', '')
    example = '{"answer":0}' if case['grader'] == 'integer' else '{"answer":["R0"]}'
    return ('Solve the task. Return only a JSON object with exactly one key "answer". '
            'Use the value type in this example (the example is not the solution): '
            + example + '. No explanations or extra keys.\nTask:\n' + task)


def normalize_answer(answer, case):
    try:
        obj = json.loads(answer)
        if not isinstance(obj, dict) or set(obj) != {'answer'}:
            return ''
        value = obj['answer']
        if case['grader'] == 'integer':
            if type(value) is not int:
                return ''
        elif not isinstance(value, list) or not all(type(v) is str for v in value):
            return ''
        return json.dumps(value)
    except (ValueError, TypeError):
        return ''
