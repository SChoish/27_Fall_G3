"""Count the incremental transmitter energy of the three calibration payloads."""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from experiments.analyze_phase_feedback import load_episodes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episodes', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    protocol, episodes = load_episodes(args.episodes)
    grouped = defaultdict(list)
    for episode in episodes:
        probe = sum(p['energy']*p['bits'] for p in episode['probe'])
        validation = sum(p['energy']*p['bits'] for p in episode['validation'])
        bits = sum(p['bits'] for p in episode['validation'])
        grouped[episode['design']].append([probe, validation, bits])
    lines = ['# 세 위상 관측에 쓰는 송신 에너지', '',
             f"채널 seed {protocol['root_seed']}, 각 설계당 {protocol['channels']}채널이다. TX 전류를 패킷 payload 구간에서 적분했다. 물리 측정으로 얻은 배선당 에너지를 세 배선에 합산했다. 3점 보정 계열은 같은 훈련 패킷을 사용하며 고정 위상의 운영 보정 비용은 0이다.",
             '', '|설계|3회 보정 TX pJ/링크|검증 payload TX pJ/bit|보정 포함 TX pJ/검증 bit|보정 에너지 / 검증 에너지|',
             '|---|---:|---:|---:|---:|']
    for name, values in grouped.items():
        probe, validation, bits = np.mean(values, axis=0)
        lines.append(f'|{name}|{3*probe:.2f}|{validation/bits:.6f}|{(probe+validation)/bits:.6f}|{100*probe/validation:.2f}%|')
    lines += ['', '보정 포함 열은 보정을 매 채널에서 한 번 실시하고 검증 payload 네 패킷만 보내는 경우다. 더 많은 사용자 비트를 보내면 보정 기여분은 배선당 전송 비트 수에 반비례한다. 수신기·마진 모니터·클록 회로 및 패킷 프리앰블 에너지는 포함하지 않았다.']
    args.output.write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
