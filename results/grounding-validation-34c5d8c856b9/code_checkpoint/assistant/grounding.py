"""Evidence checks; source identity alone is not claim support."""
from __future__ import annotations

import re
import unicodedata
from decimal import Decimal, InvalidOperation
from pathlib import Path

from .config import Config, PROJECT_ROOT, fingerprint

VERSION = 'quoted-evidence-nli-v2'
IDENTIFIER = re.compile(r'(?<![\w/])([A-Za-z][A-Za-z0-9]*(?:[-._/][A-Za-z0-9]+)*(?:\s+\d+(?:\.\d+)*(?:\s+\d+(?:\.\d+)?[BbMm])?)?)(?!\w)')
QUOTED = re.compile(r'["“]([^"”\n]{2,80})["”]')
NUMBER = re.compile(r'(?<![\w.])[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:e[-+]?\d+)?(?!\w)', re.I)


def normalized(text: str) -> str:
    return ' '.join(unicodedata.normalize('NFKC', text).split())


def quotation_span(text: str, quote: str) -> tuple[int, int] | None:
    def view(value):
        wrapped = {i for m in re.finditer(r'(?<=\w)-[ \t]*\r?\n[ \t]*(?=\w)', value)
                   for i in range(m.start(), m.end())}
        chars, positions = [], []
        for i, char in enumerate(value):
            if i in wrapped or char == '\u00ad':
                continue
            for item in unicodedata.normalize('NFKC', char):
                if not item.isspace():
                    chars.append(item); positions.append(i)
        return ''.join(chars), positions
    haystack, positions = view(text)
    needle, _ = view(quote)
    start = haystack.find(needle) if needle else -1
    return (positions[start], positions[start+len(needle)-1]+1) if start >= 0 else None


def compact(text: str) -> str:
    return ''.join(c for c in normalized(text).casefold() if c.isalnum())


def identifiers(text: str) -> list[str]:
    # High-confidence literal identifiers only, not arbitrary capitalized nouns.
    values = []
    for match in IDENTIFIER.finditer(text):
        value = match.group(1)
        if not any(c.isdigit() for c in value):
            continue
        if any(c.isdigit() for c in value.split()[0]):
            values.append(value)
        elif any(c.isupper() for c in value.split()[0]) or '.' in value:
            # A standalone four-digit year is a date, not a model identifier.
            if not re.fullmatch(r'[A-Za-z]+\s+(?:19|20)\d{2}', value):
                values.append(value)
    values += QUOTED.findall(text)
    return list(dict.fromkeys(values))


def contains_identifier(text: str, identifier: str) -> bool:
    target = compact(identifier)
    # Boundaries prevent e.g. GPT-4 from matching GPT-40. Spaces and separators
    # inside an identifier may vary, but text on either side must not be alphanumeric.
    pattern = r'(?<![\w])' + r'[\W_]*'.join(re.escape(c) for c in target) + r'(?![\w])'
    return bool(re.search(pattern, normalized(text).casefold()))


def question_coverage(question: str, sources: list[dict]) -> dict:
    requested = identifiers(question)
    missing = [term for term in requested if not any(contains_identifier(s['text'], term) for s in sources)]
    return {'version': VERSION, 'requested_identifiers': requested, 'missing_identifiers': missing,
            'passed': not missing, 'scope': 'retrieved passage text; not a claim about the entire paper'}


def numbers(text: str) -> set[str]:
    result = set()
    for match in NUMBER.finditer(normalized(text)):
        try:
            result.add(str(Decimal(match.group().replace(',', '')).normalize()))
        except InvalidOperation:
            continue
    return result


def model_cache(config: Config, root: Path = PROJECT_ROOT) -> Path:
    return root / '.cache/grounding' / fingerprint({'model': config.grounding_model,
                                                   'revision': config.grounding_revision})[:20]


class EntailmentModel:
    def __init__(self, config: Config, root: Path = PROJECT_ROOT):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        path = model_cache(config, root)
        if not (path / 'verified.json').exists():
            raise RuntimeError('Claim-verification model is not prepared. Run scripts/prepare_grounding.py first.')
        torch.set_num_threads(config.cpu_threads)
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True,
                                                                       use_safetensors=True).eval()
        self.labels = {int(k):v.casefold() for k,v in self.model.config.id2label.items()}
        if set(self.labels.values()) != {'contradiction', 'entailment', 'neutral'}:
            raise ValueError('Unexpected NLI label mapping')
        self.config = config
        self.identity = {'model': config.grounding_model, 'revision': config.grounding_revision,
                         'parameters': sum(p.numel() for p in self.model.parameters()), 'labels': self.labels}

    def score(self, pairs: list[tuple[str, str]]) -> list[dict]:
        import torch
        if not pairs:
            return []
        def nli_text(text):
            # Only the classifier view joins PDF line-wrap hyphenation. Stored
            # source spans and literal-quotation checks retain the original text.
            return normalized(re.sub(r'(?<=\w)-\s*\n\s*(?=\w)', '', text))
        pairs = [(nli_text(p),nli_text(h)) for p,h in pairs]
        lengths = [len(self.tokenizer(p, h, truncation=False)['input_ids']) for p,h in pairs]
        if any(n > 512 for n in lengths):
            raise ValueError('Verification pair exceeds 512 tokens; evidence was not silently truncated')
        result = []
        with torch.inference_mode():
            for start in range(0, len(pairs), 8):
                batch = pairs[start:start+8]
                encoded = self.tokenizer([p for p,h in batch], [h for p,h in batch],
                    padding=True, truncation=False, return_tensors='pt')
                values = self.model(**encoded).logits.softmax(dim=-1)
                if not torch.isfinite(values).all():
                    raise RuntimeError('NLI model returned nonfinite probabilities')
                probabilities = values.cpu().tolist()
                for row in probabilities:
                    result.append({label: float(row[index]) for index,label in self.labels.items()})
        return result


class GroundingVerifier:
    def __init__(self, config: Config, root: Path = PROJECT_ROOT, model=None):
        self.config, self.root, self._model = config, root, model

    @property
    def model(self):
        if self._model is None:
            self._model = EntailmentModel(self.config, self.root)
        return self._model

    def verify(self, parsed: dict, sources: list[dict]) -> dict:
        by_label = {s['label']:s for s in sources}
        checks, pairs, owners = [], [], []
        claims = parsed.get('claims', [])
        if not claims:
            return {'version': VERSION, 'passed': False, 'errors': ['Answered claims require exact supporting quotations'], 'claims': []}
        for index, claim in enumerate(claims):
            check = {'claim_index': index, 'passed': False, 'errors': [], 'evidence': []}
            evidence = claim.get('evidence', [])
            labels = set(claim.get('source_ids', []))
            if len(labels) != 1 or len(evidence) != 1:
                check['errors'].append('Split facts into atomic claims, each with one cited passage and one quotation')
            if not evidence or {e.get('source_id') for e in evidence} != labels:
                check['errors'].append('Every cited source needs a supporting quotation')
            for item in evidence:
                label, quote = item.get('source_id'), item.get('quote', '')
                source = by_label.get(label)
                if not source or not isinstance(quote, str) or not 8 <= len(quote.strip()) <= 400:
                    check['errors'].append('Invalid or empty supporting quotation')
                    continue
                span = quotation_span(source['text'], quote)
                if span is None:
                    check['errors'].append(f'Quotation must appear verbatim in {label}')
                    continue
                check['evidence'].append({'source_id': label, 'quote': source['text'][span[0]:span[1]],
                    'generated_quote':quote,'start_char':source.get('start_char',0)+span[0],
                    'end_char':source.get('start_char',0)+span[1],
                    'chunk_id': source['chunk_id'], 'paper_id': source['paper_id'],
                    'page_number': source['page_number']})
            if not check['errors']:
                combined = '\n'.join(e['quote'] for e in check['evidence'])
                missing = [x for x in identifiers(claim['text']) if not contains_identifier(combined, x)]
                missing_numbers = numbers(claim['text']) - numbers(combined)
                if missing:
                    check['errors'].append('Claim introduces an identifier absent from its quotations')
                if missing_numbers:
                    check['errors'].append('Claim introduces a number absent from its quotations')
                if not check['errors']:
                    # Verify the quotations and their parent passages separately.
                    # The second check guards against a quote omitting a negation.
                    label = next(iter(labels))
                    title = by_label[label].get('title', '')
                    prefix = f'From the paper "{title}": ' if title else ''
                    for premise in (prefix+combined, prefix+by_label[label]['text']):
                        pairs.append((premise, claim['text']))
                        owners.append(index)
            checks.append(check)
        if pairs:
            try:
                scores = self.model.score(pairs)
            except ValueError as exc:
                # An over-budget combined comparison is unsupported, not truncated.
                if '512 tokens' not in str(exc):
                    raise
                for index in set(owners):
                    checks[index]['errors'].append('Supporting evidence exceeds the verification budget')
                scores = []
            for index, score in zip(owners, scores):
                checks[index].setdefault('nli', []).append(score)
                position = len(checks[index]['nli'])
                if position == 1 and score['entailment'] < self.config.grounding_entailment_threshold:
                    checks[index]['errors'].append('The supporting quotation does not sufficiently entail the claim')
                if position == 2 and score['contradiction'] >= self.config.grounding_contradiction_threshold:
                    checks[index]['errors'].append('The parent passage contradicts the claim')
        for check in checks:
            check['passed'] = not check['errors'] and len(check.get('nli', [])) == 2
        errors = [f"Claim {c['claim_index']+1}: {error}" for c in checks for error in c['errors']]
        return {'version': VERSION, 'passed': bool(checks) and all(c['passed'] for c in checks),
                'threshold': self.config.grounding_entailment_threshold, 'errors': errors,
                'contradiction_veto':self.config.grounding_contradiction_threshold,
                'claims': checks, 'model': getattr(self._model, 'identity', None)}
