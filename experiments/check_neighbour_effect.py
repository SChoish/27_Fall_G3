"""Compare one versus two neighbours at identical speed, packets, sizing and coupling per edge."""
import argparse
from pathlib import Path
import json

from experiments.d2d_three_lane import d, make_case
from experiments.run_three_lane import CALIBRATION


def compare(calibration, output):
    case = make_case(2.15, .085, 12067, 64, True)
    full = {r['design']:r for r in calibration['rows'] if r['rate']==2.15 and r['coupling']==.085}
    evaluator = d.Evaluator([case], output/'two_lane')
    result = []
    for key, x in CALIBRATION.items():
        single = evaluator.evaluate(x)['cases'][0]
        double = full[key]
        result.append(dict(design=key, x=x, one_neighbour_margin=single['margins'][0],
                           two_neighbour_margin=double['margins'][0],
                           one_neighbour_pass=single['feasible'], two_neighbour_pass=double['feasible'],
                           one_neighbour_errors=single['errors'][0], two_neighbour_errors=double['errors'][0]))
    d.write_json(output/'neighbour_effect.json',dict(rate=2.15,coupling_pf_per_edge=.085,packet_seed=12067,
                 bits=64,rows=result,accounting=evaluator.accounting()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--calibration',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    compare(json.loads(a.calibration.read_text()),a.output)
