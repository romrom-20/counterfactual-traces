"""Build the fixed fresh literals and a hash catalog; never run the model."""

import argparse
import hashlib
import json
import re
from pathlib import Path

from transformers import AutoTokenizer

HERE=Path(__file__).resolve().parent
FAMILIES=(("cobalt","cobalt atrium","saffron gallery","Friday","Sunday"),
          ("willow","willow observatory","copper pavilion","Tuesday","Thursday"))


def digest(ids):
    return hashlib.sha256(json.dumps(ids,separators=(",",":")).encode()).hexdigest()


def main(history):
    cases_path=HERE/"cases228.json"
    catalog_path=HERE/"historical-input-hashes-228.json"
    assert not cases_path.exists() and not catalog_path.exists()
    settings=json.loads((HERE/"settings.json").read_text())["model"]
    tok=AutoTokenizer.from_pretrained(settings["model"],revision=settings["revision"],local_files_only=True)
    catalog=set()
    provenance=[]

    def scan(value):
        if isinstance(value,dict):
            for key,item in value.items():
                if key in ("free_input_ids","false_ids","true_ids") and isinstance(item,list) and len(item)>10 and all(isinstance(v,int) for v in item):
                    catalog.add(digest(item))
                else:
                    scan(item)
        elif isinstance(value,list):
            for item in value:
                scan(item)

    paths=[HERE/"cases.json"]
    paths+=sorted((history/"runs").glob("**/forecasts/*.json"))
    paths+=sorted((history/"runs").glob("**/fresh-inputs.json"))
    for p in paths:
        raw=p.read_bytes()
        before=len(catalog)
        scan(json.loads(raw))
        if len(catalog)!=before:
            provenance.append(dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest()))
    original=json.loads((HERE/"cases.json").read_text())
    jobs=[]
    fresh=set()
    for name,a,b,day0,day1 in FAMILIES:
        substitutions=dict(marzipan=a,nougat=b,Monday=day0,Wednesday=day1)
        def replace(text):
            return re.sub(r"\b(marzipan|nougat|Monday|Wednesday)\b",lambda m:substitutions[m.group()],text)
        for case in original:
            if case["group"]["block"]!="marzipan":
                continue
            ids={bit:tok.encode(replace(tok.decode(case[bit+"_ids"])),add_special_tokens=False) for bit in ("false","true")}
            assert len(ids["false"])==len(ids["true"])
            different=[i for i,(x,y) in enumerate(zip(ids["false"],ids["true"])) if x!=y]
            assert len(different)==1
            pos=different[0]
            assert ids["false"][pos]==895 and ids["true"][pos]==830
            hashes=[digest(ids[bit]) for bit in ("false","true")]
            assert not (set(hashes)&catalog) and not (set(hashes)&fresh)
            fresh.update(hashes)
            group={**case["group"],"block":name}
            key=hashlib.sha256(json.dumps(group,sort_keys=True).encode()).hexdigest()
            jobs.append(dict(key=key,group=group,position=pos,false_ids=ids["false"],true_ids=ids["true"],
                messages=[{**m,"content":replace(m["content"])} for m in case["messages"]],input_hashes=hashes,
                template_key=case["key"]))
    assert len(jobs)==32 and len(fresh)==64
    catalog_path.write_text(json.dumps(dict(hash_definition="SHA256 of compact JSON token list",hashes=sorted(catalog),
        provenance=provenance,scope="Available paired forecast records, fresh-input manifests and published225b inputs."),indent=2)+"\n")
    cases_path.write_text(json.dumps(jobs,indent=2)+"\n")
    print("Built32 pairs;64 input hashes absent from",len(catalog),"recorded historical hashes.")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-root",type=Path,required=True)
    main(parser.parse_args().historical_root.resolve())
