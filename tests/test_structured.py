import unittest
from app.schemas.chat import ChatRequest
from app.providers.openai import OpenAIProvider
from app.providers.deepseek import DeepSeekProvider
from app.core.config import ModelConfig
from experiments.structured_answers import structured_prompt, normalize_answer
from experiments.deepseek_cases import grade, make_cases

class StructuredTests(unittest.TestCase):
    def test_forward_and_default(self):
        for provider in (OpenAIProvider, DeepSeekProvider):
            body=ChatRequest(model='budget',messages=[],response_format={'type':'json_object'})
            payload=provider._build_request(None,body,ModelConfig(name='test',routing_keys=['budget']))
            self.assertEqual(payload['response_format'],{'type':'json_object'})
            body.response_format=None
            self.assertNotIn('response_format',provider._build_request(None,body,ModelConfig(name='test',routing_keys=['budget'])))
    def test_strict_contract_and_no_reference_leak(self):
        case={'prompt':'Compute 2+3.', 'grader':'integer','expected':5}
        self.assertEqual(structured_prompt(case),structured_prompt(dict(case,expected=999)))
        self.assertEqual(grade(normalize_answer('{"answer":5}',case),case),1)
        for answer in ('{"answer":true}','{"answer":"5"}','{"answer":5,"why":"x"}','5','bad'):
            self.assertEqual(normalize_answer(answer,case),'')
    def test_generated_bounds(self):
        for c in make_cases(7023,6,'train')+make_cases(7024,3,'test'):
            self.assertLessEqual(len(structured_prompt(c).encode()),1800)
