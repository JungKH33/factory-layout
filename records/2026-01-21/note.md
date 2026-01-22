# 제약 조건 추가
weight:
공장 전체에 대해 하중 임계값 weight_limit을 두고, 설비의 facility_weight가 이를 초과하는 경우에는 지정된 weight_areas 중 하나에 설비가 완전히 포함되어야 배치 가능하다. 즉 무거운 설비는 하중 허용 구역 밖으로 조금이라도 벗어나면 배치 불가로 처리한다.

height:
공장 공간은 height_areas로 덮여 각 위치의 ceiling_height가 정의된 것처럼 동작하며, 존 밖은 기본값 env.ceiling_height를 적용한다. 설비의 facility_height가 어떤 위치의 ceiling_height보다 큰 경우 그 위치와는 “겹치면 안 되는” 공간 분리 제약이 발생하므로, 해당 영역과의 overlap이 발생하는 배치는 불가로 처리한다. (즉, facility_height > ceiling_height인 영역은 금지 영역처럼 취급)

dry:
dry_areas는 사각형 존과 존별 value로 정의된다. 설비가 facility_dry 값을 가지는 경우, facility_dry <= value를 만족하는 “적합 존” 중 하나에 설비가 완전히 포함되어야 한다. 반대로 facility_dry > value인 부적합 존과는 설비가 겹치면 안 된다.
(※ 부등호 방향/정확 규칙은 최종 스펙 재확인 필요. ToDo: dry 비교 규칙(<=, >=) 확정 후 문구/구현 동기화)

clearance:
추후 추가 예정(현재는 미적용).

![zone_maps](./zone_maps.png) 

이러한 제약을 도입하면 설비마다 실제로 배치 가능한 영역의 크기가 크게 달라진다. 기존처럼 면적이 큰 설비부터 먼저 배치하는 규칙은 이 차이를 반영하지 못해, 배치가 어려운 설비가 뒤로 밀리면서 실패 가능성이 커지거나 탐색 비용이 증가할 수 있다. 따라서 초기 배치 순서를 큰 설비 우선에서 어려운 설비 우선으로 변경하였다. 각 설비에 대해 정적 금지 영역과과 설비 제약으로 인해 추가로 금지되는 영역을 합친 invalid map을 기준으로, 해당 footprint가 들어갈 수 있는 top-left 위치의 성공 횟수(K)와 전체 시도 횟수(T)를 계산해 placeable_ratio = K/T가 작은 설비부터 먼저 배치한다. 이 계산은 설비 수가 많아도 빠르게 수행되어야 하므로, footprint 크기의 커널을 이용한 2D convolution으로 overlap을 한 번에 계산하여 유효 top-left 개수를 효율적으로 산출한다.

Topk 알고리즘 수정 (가 아니라 Near Sampling 추가)
너무 제약조건이 많아져서. 
제약된 곳 근처 점 샘플링.
Near + coarse (upper limit 기본은 None 없으면. 개수 자유롭게 가능한지 검토하고, 있으면 개수만큼 채우고 마스크 씌우든지)
rotation도 고려?
처음 배치할때는 제약조건 (워닝으로만 띄우고 에러 내지는 마)