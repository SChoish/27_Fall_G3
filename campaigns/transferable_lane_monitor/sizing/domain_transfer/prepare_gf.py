"""Extract GF180MCU typical 3.3 V BSIM bins and noise parameters without editing coefficients."""
import argparse
import hashlib
from pathlib import Path
import re

def extract(source):
    expected="561e63453512eb8c614d7d45ac1c6cc2c41e0111"
    raw=Path(source).read_bytes()
    assert hashlib.sha1(f"blob {len(raw)}\0".encode()+raw).hexdigest()==expected
    s=raw.decode().replace("\r","")
    parts=[s[:s.find("\n",s.find("limitations under"))+1], "* TT 3.3 V raw four-terminal MOS models; no mismatch or SAB extension",
           ".param fnoicor=0 sw_stat_global=0 sw_stat_mismatch=0"]
    for name in ("noise_corner","nmos_3p3_t","pmos_3p3_t"):
        block=re.search(r"^\.lib "+name+r"\s*\n(.*?)^\.endl",s,re.I|re.M|re.S).group(1)
        if name!="noise_corner":
            block=block[re.search(r"^\.model\s+",block,re.I|re.M).start():]
        parts.append(block)
    return "\n".join(parts)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("source")
    p.add_argument("dest",type=Path)
    args=p.parse_args()
    args.dest.write_text(extract(args.source))
