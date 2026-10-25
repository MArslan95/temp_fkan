"""Source-level release checks; pending experimental evidence is not a TeX error."""
import json
import re
from collections import Counter
from pathlib import Path
from .protocol import sha256, write_json

ROOT=Path(__file__).resolve().parents[1]


def validate():
    errors=[]
    paper=ROOT/'paper'
    response=(paper/'repose_filled.tex').read_text(encoding='utf-8')
    source=(paper/'highlighted_filled.tex').read_text(encoding='utf-8')
    clean=(paper/'clean_filled.tex').read_text(encoding='utf-8')
    expected={f'R1-C{i}' for i in range(1,6)}|{f'R2-C{i}' for i in range(1,10)}
    if set(re.findall(r'CONCERN-ID: (R\d-C\d+)',response))!=expected:
        errors.append('Not all 14 reviewer concerns are present')
    if len(re.findall(r'\\textbf\{Author response:\}',response))!=14 or len(re.findall(r'\\textbf\{Author action:\}',response))!=14:
        errors.append('Expected 14 response/action pairs')
    normalized=source.replace('\\revisionhighlightstrue','\\revisionhighlightsfalse',1)
    normalized=normalized.replace('\\colorbox{yellow}{Revised operation diagram.}','Revised operation diagram.')
    if clean!=normalized:
        errors.append('Clean and highlighted scientific content differ')
    combined=source
    for target in re.findall(r'\\input\{([^{}]+)\}',source):
        path=paper/(target if Path(target).suffix else target+'.tex')
        if not path.exists():
            errors.append('Missing input: '+target)
        else:
            text=path.read_text(encoding='utf-8')
            combined+='\n'+'\n'.join(line for line in text.splitlines() if not line.lstrip().startswith('%'))
    labels=re.findall(r'\\label\{([^{}]+)\}',combined)
    errors+=['Duplicate active label: '+key for key,count in Counter(labels).items() if count>1]
    refs=re.findall(r'\\(?:ref|eqref)\s*\{([^{}]+)\}',combined)
    errors+=['Undefined reference: '+key for key in sorted(set(refs)-set(labels))]
    for dataset in ('HC','HH','PI','TH','QUH'):
        if combined.count('\\label{fig:'+dataset+'}')!=1:
            errors.append('Missing/repeated dataset map: '+dataset)
    for target in re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^{}]+)\}',combined):
        if not (paper/target).is_file():
            errors.append('Missing figure: '+target)
    bib=(paper/'revision_assets/support/sample.bib').read_text(encoding='utf-8')
    keys=set(re.findall(r'@\w+\s*\{\s*([^,\s]+)',bib))
    for cited in re.findall(r'\\cite\{([^{}]+)\}',combined):
        errors+=['Missing citation key: '+key for key in cited.split(',') if key.strip() not in keys]
    manifest=json.loads((ROOT/'revision/reference/manifest.json').read_text())
    for name,digest in manifest.items():
        if sha256(ROOT/name)!=digest:
            errors.append('Original source changed: '+name)
        if sha256(ROOT/'revision/reference'/name)!=digest:
            errors.append('Frozen source changed: '+name)
    if '\\textcolor{yellow}' in source:
        errors.append('Unreadable yellow-foreground revision remains')
    return dict(source_checks='pass' if not errors else 'fail',errors=errors,
                concerns=14,frozen_originals=len(manifest),
                manuscript_pending_notices=source.count('\\pending{'),
                response_pending_notices=response.count('\\pending{'),
                submission_ready=False,
                reason='Experiments, authenticated baselines, full FLOPs, external Mamba parity and release remain pending.')


if __name__=='__main__':
    result=validate()
    write_json(ROOT/'revision/document_validation.json',result)
    print(json.dumps(result,indent=2))
    if result['errors']:
        raise SystemExit(1)
