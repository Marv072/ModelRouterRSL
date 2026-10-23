import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import AsyncMock, patch

from app.core.config import ProviderConfig, ApiKeyConfig, ModelConfig
from app.core.rsl import classify
from app.providers.openai import OpenAIProvider
from app.schemas.chat import ChatRequest, ChatMessage, ChatResponse, ChatChoice, Usage
from experiments.deepseek_cases import make_cases, grade
from experiments.run_deepseek import BudgetProvider


class DatasetTests(unittest.TestCase):
    def test_split_labels_and_input_bounds(self):
        train, test = make_cases(6023,16,"train"), make_cases(6024,8,"test")
        self.assertEqual((len(train),len(test)),(48,24))
        self.assertFalse({c['prompt'] for c in train} & {c['prompt'] for c in test})
        for case in train+test:
            self.assertLessEqual(len(case['prompt'].encode()),1800)
            self.assertEqual(classify([{'role':'user','content':case['prompt']}]),case['category'])
            answer = json.dumps(case['expected'])
            self.assertEqual(grade(answer,case),1)
            self.assertEqual(grade('unrelated answer',case),0)
            if case['category'] == 'coding':
                code = case['prompt'].split('\n',1)[1]
                out = io.StringIO()
                # Only execute our static generator's trusted snippet, never model output.
                with redirect_stdout(out):
                    exec(compile(code,'generated-fixture','exec'),{})
                self.assertEqual(int(out.getvalue()),case['expected'])


class BudgetTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        Path('.test-tmp').mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir='.test-tmp')
        self.ledger = Path(self.tmp.name)/'budget.json'
        self.ledger.write_text(json.dumps(dict(budget_cny=5,reserved_cny=0.04,calls=[])),encoding='utf-8')
        self.provider = BudgetProvider(ProviderConfig(base_url='https://example.test',max_retries=0,
            keys=[ApiKeyConfig(key='test',rpm=0)],models=[]),self.ledger)
        self.body = ChatRequest(model='budget',messages=[ChatMessage(role='user',content='hello')],
                                thinking={'type':'disabled'},max_tokens=256)
        self.model = ModelConfig(name='deepseek-flash',routing_keys=['budget'])

    async def asyncTearDown(self):
        await self.provider.client.aclose()
        self.tmp.cleanup()

    async def test_thinking_forwarded_and_cache_usage_preserved(self):
        payload = self.provider._build_request(self.body,self.model)
        self.assertEqual(payload['thinking'],{'type':'disabled'})
        usage = Usage(prompt_tokens=10,completion_tokens=1,total_tokens=11,prompt_cache_hit_tokens=4)
        self.assertEqual(usage.model_dump()['prompt_cache_hit_tokens'],4)

    async def test_success_reserves_before_call(self):
        response = ChatResponse(id='x',created=0,model='deepseek-flash',choices=[ChatChoice(
            message=ChatMessage(role='assistant',content='OK'))])
        with patch.object(OpenAIProvider,'chat_completions',new=AsyncMock(return_value=response)) as call:
            await self.provider.chat_completions(self.body,self.model)
            call.assert_awaited_once()
        self.assertAlmostEqual(json.loads(self.ledger.read_text())['reserved_cny'],0.07)

    async def test_failure_keeps_reservation_and_never_retries(self):
        with patch.object(OpenAIProvider,'chat_completions',new=AsyncMock(side_effect=RuntimeError('fake'))) as call:
            with self.assertRaises(RuntimeError):
                await self.provider.chat_completions(self.body,self.model)
            call.assert_awaited_once()
        self.assertAlmostEqual(json.loads(self.ledger.read_text())['reserved_cny'],0.07)

    async def test_budget_exhaustion_blocks_upstream(self):
        self.ledger.write_text(json.dumps(dict(budget_cny=5,reserved_cny=4.99,calls=[])))
        with patch.object(OpenAIProvider,'chat_completions',new=AsyncMock()) as call:
            with self.assertRaises(ValueError):
                await self.provider.chat_completions(self.body,self.model)
            call.assert_not_awaited()

    async def test_oversized_or_thinking_request_blocked(self):
        with patch.object(OpenAIProvider,'chat_completions',new=AsyncMock()) as call:
            for change in ({'max_tokens':1000},{'thinking':None},
                           {'messages':[ChatMessage(role='user',content='x'*1801)]}):
                with self.assertRaises(ValueError):
                    await self.provider.chat_completions(self.body.model_copy(update=change),self.model)
            call.assert_not_awaited()
