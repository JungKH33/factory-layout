# Greedy + Search 실험

위에서 말한 Top-K 샘플링에 greedy policy를 통해 결과가 얼마나 잘 나오는지 확인하였습니다. Greedy란 딱 한 수 앞만을 고려해서 비용을 가장 크게 줄이는 행동을 선택하는 정책을 의미합니다. 다만 greedy policy만을 적용하면 local minima에 빠지기 쉬워 물류와 같은 설비 간 관계를 충분히 고려하기 어렵고, 특히 좁은 공간에 배치해야 하는 경우 이후 배치 가능성을 남겨두지 못해 배치가 실패하는 상황도 발생할 수 있습니다.

따라서 본 실험에서는 greedy에 탐색 알고리즘을 결합했을 때 실제 성능 개선으로 이어지는지 확인해보았습니다. Greedy 단독과 greedy + search의 성능을 비교하였고, search 알고리즘으로는 MCTS와 Beam Search를 사용하였습니다.

MCTS는 현재 상태에서 가능한 행동들을 트리 형태로 확장하면서 다수의 시뮬레이션을 통해 각 행동의 장기적인 기대 성능을 추정하는 탐색 기법입니다. 여러 단계 이후의 결과까지 반영할 수 있어, 초반 선택이 이후 배치 가능성을 크게 좌우하는 복잡한 문제에서 배치 안정성을 높이는 데 유리합니다. 다만 계산 비용이 크고, 시뮬레이션 수나 깊이가 충분하지 않으면 추정이 불안정해져 최선의 해를 찾지 못하거나 오히려 greedy보다 성능이 떨어질 수 있습니다. 대신 배치 실패가 자주 발생하는 복잡한 문제에서 안정적으로 배치를 완성하는 방향으로 탐색이 유도되는 경향이 있습니다.

Beam Search는 각 단계에서 가능한 후보 중 상위 몇 개만 유지한 채 다음 단계로 확장하는 제한 탐색 방식입니다. 유망한 후보를 여러 개 유지하며 진행하기 때문에 한 번의 선택 실수로 이후에 배치가 막히는 상황을 일부 완화할 수 있습니다. 또한 MCTS처럼 많은 시뮬레이션을 수행하지 않기 때문에 계산 비용이 상대적으로 작고, greedy와 유사한 구조라 구현 및 튜닝 부담도 비교적 낮습니다. 다만 현재 점수 기준으로 상위 후보 중심 탐색을 진행하므로 장기적으로는 유리하지만 당장 점수가 낮은 경로를 일찍 버릴 수 있고, 결과적으로 best 위주로 편향되면 greedy와 큰 차이를 만들지 못할 가능성이 있습니다.

제약조건이 많지 않은 쉬운 문제의 경우 greedy와 greedy + MCTS 모두 비슷한 성능을 보여주었습니다.

| Greedy (cost = 1040.5) | Greedy + MCTS (cost = 1074.5) | Greedy + Beam Search (cost = 1066.0) |
|---|---|---|
| ![vanilla](./vanilla.png) | ![mcts](./mcts.png) | ![beam](./beam.png) |


하지만 어려운 문제에서는 greedy + MCTS의 점수가 크게 앞서는 경우가 많았고, 다른 알고리즘이 배치에 실패한 상황에서도 배치에 성공하는 사례를 확인할 수 있었습니다.

| Greedy (cost = 1378.5) | Greedy + MCTS (cost = 1127.0) | Greedy + Beam Search (cost = 1272) |
|---|---|---|
| ![vanilla_hard](./vanilla_hard.png) | ![mcts_hard](./mcts_hard.png) | ![greedy_beam_hard](./beam_hard.png) |

| Greedy (배치실패) | Greedy + MCTS (cost = 1329.0) | Greedy + Beam Search (배치실패) |
|---|---|---|
| ![vanilla_hard2](./vanilla_hard2.png) | ![mcts_hard2](./mcts_hard2.png) | ![greedy_beam_hard2](./beam_hard2.png) |






