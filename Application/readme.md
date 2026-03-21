# Exploration geophysics
## Seismic survey (지진파 탐색)
### Full-waveform inversion
- Full-waveform inversion, Part 1: Forward modeling : Time sereis Data Generation, 유한차분법 (Finite-Difference Method, FDM), 시간 단계별 계산 (Time-stepping), 흡수 경계 조건 (Absorbing Boundary Conditions, ABC), Acoustic Wave Equation

#### Libraries
- Devito: 파동 방정식 시뮬레이션을 위해 고도로 최적화된 C 코드를 자동으로 생성하는 도메인 특화 언어(DSL) 기반의 Python 프레임워크입니다. : https://www.devitoproject.org/ , https://github.com/devitocodes/devito
- SymPy: 파동 방정식을 기호적(Symbolic) 표현식으로 정의하여 미분 및 이산화 과정을 수식적으로 처리하는 데 사용됩니다.

#### Reference
- https://github.com/seg/tutorials-2017 : Code
