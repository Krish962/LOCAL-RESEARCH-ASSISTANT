"""Run with the isolated evaluator: python -m unittest evaluation.test_local_judge."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from pydantic import BaseModel, ValidationError
from assistant.config import PROJECT_ROOT, configure_cache
configure_cache()
from .local_judge import OllamaStructuredJudge


class Statements(BaseModel):
    statements: list[str]


class JudgeTransportTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=PROJECT_ROOT / '.cache')
        self.output = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.client = SimpleNamespace(chat=AsyncMock())
        self.judge = OllamaStructuredJudge('llama3.2:3b', 'http://127.0.0.1:11434',
                                          self.output, client=self.client)

    async def test_schema_is_sent_to_native_api_and_validated(self):
        self.client.chat.return_value = SimpleNamespace(model_dump=lambda **kw: {
            'done_reason': 'stop', 'message': {'content': '{"statements":["A supported claim."]}'}})
        result = await self.judge.agenerate('Evaluate this answer.', Statements)
        self.assertEqual(result.statements, ['A supported claim.'])
        self.assertEqual(self.client.chat.call_args.kwargs['format'], Statements.model_json_schema())
        self.assertEqual(json.loads(next(self.output.glob('*.json')).read_text())['status'], 'completed')

    async def test_schema_echo_is_an_error_not_a_score(self):
        self.client.chat.return_value = SimpleNamespace(model_dump=lambda **kw: {
            'message': {'content': json.dumps(Statements.model_json_schema())}})
        with self.assertRaises(ValidationError):
            await self.judge.agenerate('Evaluate.', Statements)
        self.assertEqual(json.loads(next(self.output.glob('*.json')).read_text())['status'], 'failed')

    async def test_oversized_task_is_not_silently_truncated(self):
        with self.assertRaises(ValueError):
            await self.judge.agenerate('x' * 17000, Statements)
        self.client.chat.assert_not_called()

    def test_remote_endpoint_is_rejected(self):
        with self.assertRaises(ValueError):
            OllamaStructuredJudge('model', 'https://example.com', self.output, client=self.client)


if __name__ == '__main__':
    unittest.main()
