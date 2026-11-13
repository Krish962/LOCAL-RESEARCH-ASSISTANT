"""Source-bound extractive fallback: choose IDs, never generate research facts."""
from __future__ import annotations
import re
from .grounding import normalized, quotation_span


def excerpt_text(text):
    return normalized(re.sub(r'(?<=\w)-\s*\n\s*(?=\w)', '',text))


def span_overlap(a,b):
    if (a['paper_id'],a['page_number'])!=(b['paper_id'],b['page_number']):return 0.
    overlap=max(0,min(a['end_char'],b['end_char'])-max(a['start_char'],b['start_char']))
    return overlap/max(1,min(a['end_char']-a['start_char'],b['end_char']-b['start_char']))


def units(question, sources, reranker, limit=10, required_papers=()):
    candidates=[]
    for source in sources:
        text=source['text']
        starts=[0]
        for match in re.finditer(r'[.!?]\s+(?=[A-Za-z])',text):
            prefix=text[max(0,match.start()-8):match.start()+1].lower()
            if any(prefix.endswith(a) for a in ['e.g.','i.e.','et al.','fig.','sec.','dr.','vs.']):continue
            starts.append(match.end())
        ends=starts[1:]+[len(text)]
        spans=[]
        for start,end in zip(starts,ends):
            while end-start>400:
                boundary=text.rfind(' ',start+200,start+400)
                if boundary<=start:boundary=start+400
                spans.append((start,boundary))
                start=max(start+1,boundary-60)
                while start<end and not text[start].isspace():start+=1
            spans.append((start,end))
        for a,b in spans:
            while a<b and text[a].isspace():a+=1
            while b>a and text[b-1].isspace():b-=1
            if not 12<=b-a<=400:continue
            quote=text[a:b]
            if len(normalized(quote).split())<4:continue
            candidates.append({'source_id':source['label'],'quote':quote,'start_char':source.get('start_char',0)+a,
                'end_char':source.get('start_char',0)+b,'paper_id':source['paper_id'],
                'page_number':source['page_number'],'chunk_id':source['chunk_id'],'title':source['title'],
                'context':text[:160]+'\n...\n'+text[max(0,a-80):min(len(text),b+80)],'parent_context':text})
    if not candidates:return []
    ranking_texts=[]
    for c in candidates:
        value='Paper: '+c['title']+'\nContext: '+excerpt_text(c['parent_context'])+'\nExcerpt: '+excerpt_text(c['quote'])
        if hasattr(reranker,'tokenizer') and len(reranker.tokenizer(question,value,truncation=False)['input_ids'])>512:
            value='Paper: '+c['title']+'\nExcerpt: '+excerpt_text(c['quote'])
        ranking_texts.append(value)
    scores,_=reranker.scores(question,ranking_texts)
    ordered=sorted(zip(candidates,scores),key=lambda x:(-x[1],x[0]['chunk_id'],x[0]['start_char']))
    required=[]
    for paper in required_papers:
        pair=next((pair for pair in ordered if pair[0]['paper_id']==paper),None)
        if pair is not None and pair not in required:required.append(pair)
    # Cover each retrieved parent before filling the pool globally. This retains
    # facts whose sentence alone lacks the method name or section heading.
    quota=[]
    for level in range(2):
        for parent in sources:
            children=[pair for pair in ordered if pair[0]['source_id']==parent['label']]
            if len(children)>level and children[level] not in required+quota:quota.append(children[level])
    ordered=required+quota+[pair for pair in ordered if pair not in required+quota]
    selected=[];seen=set()
    for candidate,score in ordered:
        key=normalized(candidate['quote'])
        if key in seen or any(span_overlap(candidate,c)>=.8 for c in selected):continue
        seen.add(key)
        selected.append({**candidate,'evidence_id':f'E{len(selected)+1}','relevance_score':score})
        if len(selected)==limit:break
    return selected


def supported_selection(data, candidates, support):
    ids=data.get('evidence_ids',[])
    if any(i not in {c['evidence_id'] for c in candidates} for i in ids):
        raise ValueError('Support selection refers to an unavailable excerpt')
    checks=support.get('claims',[])
    if len(ids)!=len(checks):raise ValueError('Support checks do not match the selected excerpts')
    kept=[identifier for identifier,check in zip(ids,checks) if check.get('passed')]
    return {'status':'answered' if kept else 'insufficient_evidence','evidence_ids':kept}


def focused_review(candidates,required_papers=()):
    ranked=sorted(candidates,key=lambda c:-c['relevance_score'])
    chosen=[]
    for paper in required_papers:
        match=next((c for c in ranked if c['paper_id']==paper),None)
        if match is not None:chosen.append(match)
    chosen.extend(c for c in ranked if c not in chosen)
    return chosen[:max(2,len(required_papers))]


def render_selection(data, candidates, sources, required_papers=()):
    if not isinstance(data,dict) or set(data)!={'status','evidence_ids'}:
        raise ValueError('Evidence selection must contain only status and evidence_ids')
    if data['status']=='insufficient_evidence' and data['evidence_ids']==[]:
        return {'status':'insufficient_evidence','answer':'The retrieved excerpts do not provide the requested information.'}
    ids=data['evidence_ids']
    if (data['status']!='answered' or not isinstance(ids,list) or not 1<=len(ids)<=4
        or any(not isinstance(i,str) for i in ids) or len(set(ids))!=len(ids)):
        raise ValueError('Invalid evidence selection')
    by_id={c['evidence_id']:c for c in candidates}
    if any(i not in by_id for i in ids):raise ValueError('Selected excerpt is not available')
    chosen=[by_id[i] for i in ids]
    if set(required_papers)-{c['paper_id'] for c in chosen}:
        return {'status':'insufficient_evidence','answer':'The selected excerpts do not cover every requested paper.'}
    by_label={s['label']:s for s in sources}
    for c in chosen:
        parent=by_label[c['source_id']]
        if quotation_span(parent['text'],c['quote']) is None:
            raise RuntimeError('An extractive excerpt changed after selection')
    claims=[{'text':excerpt_text(c['quote']),'source_ids':[c['source_id']],
             'evidence':[{'source_id':c['source_id'],'quote':c['quote']}]} for c in chosen]
    return {'status':'answered','answer':'Source excerpts:\n\n'+'\n\n'.join(
        c['title']+' — PDF page '+str(c['page_number'])+'\n“'+excerpt_text(c['quote'])+'” ['+c['source_id']+']' for c in chosen),
        'answer_mode':'extractive','claims':claims,'selected_evidence':chosen}
