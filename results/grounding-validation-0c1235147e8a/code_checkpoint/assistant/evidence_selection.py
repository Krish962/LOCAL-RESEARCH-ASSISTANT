"""Source-bound extractive fallback: choose IDs, never generate research facts."""
from __future__ import annotations
import re
from .grounding import normalized, quotation_span


def excerpt_text(text):
    return normalized(re.sub(r'(?<=\w)-\s*\n\s*(?=\w)', '',text))


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
                'context':text[max(0,a-100):min(len(text),b+100)]})
    if not candidates:return []
    scores,_=reranker.scores(question,[excerpt_text(c['quote']) for c in candidates])
    ordered=sorted(zip(candidates,scores),key=lambda x:(-x[1],x[0]['chunk_id'],x[0]['start_char']))
    required=[]
    for paper in required_papers:
        pair=next((pair for pair in ordered if pair[0]['paper_id']==paper),None)
        if pair is not None and pair not in required:required.append(pair)
    ordered=required+[pair for pair in ordered if pair not in required]
    selected=[];seen=set()
    for candidate,score in ordered:
        key=normalized(candidate['quote'])
        if key in seen:continue
        seen.add(key)
        selected.append({**candidate,'evidence_id':f'E{len(selected)+1}','relevance_score':score})
        if len(selected)==limit:break
    return selected


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
