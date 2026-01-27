# Agent 비교와 평가

RL 기반반 greedy + MCTS, alphachip, maskplace 모델을 적용했습니다. 

input 맵 생성
기초적인 모델을 제작하였습니다. 
224 * 224 사이즈즈
모델의 인풋에는 3가지가 필요합니다. canvas map, invalid map, reward map


reward map의 경우 해당 위치에 놓았을때 보상 변화를 계산해야합니다.
하지만 224 * 224 (총 ) flow graph 계산할때 다익스트라를 계산하면 병렬화가 안됨.
충돌은 고려하지 않은 맨해튼 거리 + compact map으로 하면 병렬화해서 계산 가능. 


이 총 과정이 100ms 안에 이뤄지도도록 병렬화를 진행하였습니다. 


