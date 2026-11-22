"""Portable regression tests: local alongside script or repo tests/data/."""
import argparse
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
REPO=HERE.parent.parent
SCRIPT=next(p for p in [HERE/"split_uiuc.py",REPO/"scripts"/"data"/"split_uiuc.py"] if p.exists())
CONFIG=next(p for p in [HERE/"split_config.json",REPO/"data"/"splits"/"uiuc_v1"/"split_config.json"] if p.exists())
SPEC=importlib.util.spec_from_file_location("split_uiuc_under_test",SCRIPT)
M=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def poly(thickness):
    return [[0,0],[.3,-thickness*.7],[.7,-thickness*.6],[1,0],[.7,thickness*.8],[.3,thickness]]


class SplitTests(unittest.TestCase):
    def test_nonadjacent_crossing_is_rejected(self):
        errors,_=M.polygon_qa([[0,0],[1,.1],[0,.1],[1,0]],1e-12)
        self.assertIn("nonadjacent_segment_intersection_or_touch",errors)
        self.assertEqual(M.polygon_qa(poly(.1),1e-12)[0],[])

    def test_smoothing_and_naca_aliases_share_keys(self):
        config=json.loads(CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(M.name_key("davis",config),M.name_key("davis_corrected",config))
        self.assertEqual(M.name_key("a18",config),M.name_key("a18sm",config))
        self.assertEqual(M.name_key("n0012",config),M.name_key("naca0012",config))

    def test_signature_ignores_added_collinear_sampling(self):
        original=poly(.1)
        dense=[]
        for a,b in zip(original,original[1:]+original[:1]):
            dense.extend([a,[(a[0]+b[0])/2,(a[1]+b[1])/2]])
        a,_=M.shape_signature(original,129)
        b,_=M.shape_signature(dense,129)
        self.assertLess(max(abs(x-y) for x,y in zip(a,b)),1e-14)

    def test_groups_never_split_and_input_order_is_irrelevant(self):
        names="naca0012 bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa quebec romeo sierra tango uniform victor whiskey xray yankee zulu".split()
        samples=[]
        for i,name in enumerate(names):
            canonical=f"{i:064x}"
            samples.append(dict(sample_id="uiuc_"+canonical[:16],canonical_sha256=canonical,
               aliases=[name+"1"],polygon=poly(.025+i*.006),
               source_files=[dict(relative_path=name+".dat",filename=name+".dat",sha256="0"*64)]))
        # IDs must be distinct in the first 16 hex digits, as in real SHA256 IDs.
        for i,s in enumerate(samples):
            s["canonical_sha256"]=M.sha256((str(i)+"fixture").encode())
            s["sample_id"]="uiuc_"+s["canonical_sha256"][:16]
        samples[1]["polygon"]=samples[0]["polygon"] # Cross-name exact/shape duplicate.
        samples[2]["aliases"]=["naca0012_modified"] # Same parent name, different shape.
        with tempfile.TemporaryDirectory(prefix="uiuc-split-tests-") as tmp:
            tmp=Path(tmp)
            src=tmp/"samples.jsonl"
            def execute(records,output):
                src.write_text("\n".join(json.dumps(s) for s in records),encoding="utf-8")
                with redirect_stdout(io.StringIO()):
                    M.run(argparse.Namespace(input=str(src),output=str(output),config=str(CONFIG),raw_root=None))
                return {split:json.loads((output/(split+".json")).read_text()) for split in ["train","val","test"]}
            first=execute(samples,tmp/"first")
            second=execute(list(reversed(samples)),tmp/"second")
            self.assertEqual(first,second)
            membership={sid:split for split,values in first.items() for sid in values}
            self.assertEqual(len(membership),len(samples))
            self.assertEqual(membership[samples[0]["sample_id"]],membership[samples[1]["sample_id"]])
            self.assertEqual(membership[samples[0]["sample_id"]],membership[samples[2]["sample_id"]])
            self.assertEqual(membership[samples[0]["sample_id"]],"test")
            self.assertTrue(all(first.values()))


if __name__=="__main__":
    unittest.main()
