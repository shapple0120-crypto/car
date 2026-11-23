# 반응형(Reactive) 주행기: FTG + 원호 검사 (LiDAR only)

ForzaETH F1TENTH 시뮬레이터(ROS 2 Humble)용 **반응형 주행기**다. 라이다 스캔 한 장만 보고 매 순간 조향과 속도를 정한다.
팀 최종 주행기(LiDAR CorridorMPC, 계획형 proactive)와 같은 조건에서 비교하기 위한 **기준선(baseline)**이며,
"단순 반응형의 한계를 분석하고 어디까지 개선할 수 있는지"를 보여 주는 것이 목적이다.

| 주행기 | 파일 | 클래스 | 한 줄 요약 |
|---|---|---|---|
| **v3 (주력)** | `drivers/ftg_arc_driver.py` | `FTGArcDriver` | FTG가 방향을 고르고, 원호 검사로 실제로 갈 수 있는 조향·속도를 정함 |
| v2 | `drivers/ftg_driver.py` | `FTGDriver` | Follow-the-Gap + Disparity Extender (v3의 방향 고르기 부품) |

공통 사항
- 입력 `/scan`(1080빔, 시야 4.7 rad), 출력 `/drive`(조향각, 속도). 주행 중 GT(정답 위치·장애물 위치)를 쓰지 않는다.
- 지도, 레이싱라인, 맵 좌표를 쓰지 않는다 → 구간별 하드코딩 없음.
- 클래스 하나가 `/drive`를 낸다 → 여러 플래너·제어기를 고르는 MUX 없음.
- `gt_free = True` → `avoid_node.py`가 GT 토픽(`/car_state/*_GT`)을 구독하지 않는다.
- 팀 공통 인터페이스 `act(scan, x, y, yaw, v, yaw_rate, t) -> (steer, speed)`, `reset()` → 팀원 저장소의 `harness.py`, `avoid_node.py`, `run_avoid.sh`에 그대로 끼워 쓴다.

## 1. 알고리즘

### 1.1 v2: Follow-the-Gap + Disparity Extender (`FTGDriver`)
1. NaN·inf 정리한 원본 거리와 3칸 이동평균 거리를 따로 만든다.
2. 정면 기준 좌우 85°만 본다.
3. Disparity Extender: 옆 빔과 0.35 m 넘게 차이 나는 곳(장애물 모서리)에서, 먼 쪽 빔들을 차 반폭 + 여유(0.25 m)만큼 가까운 거리로 덮는다. 검사는 원본 거리로 한다.
4. 가장 가까운 점 주변을 막는다(안전 거품, 최대 30°).
5. 1.2 m보다 멀리 뚫린 연속 구간 중 가장 넓은 빈틈을 고르고, 그 안에서 깊은 쪽(최대 거리의 92% 이상) 평균 방향을 목표로 한다.
6. 목표 방향 → 조향각. 많이 꺾을수록, 정면이 가까울수록 느리게(5.5 → 1.8 m/s, 정면 0.6 m 안쪽이면 0.5 m/s).

**v2의 한계 (harness에서 확인)**: "어디가 뚫렸나"만 보고 "차가 실제로 거기까지 돌아 들어갈 수 있나"는 보지 않는다.
berlin 첫 헤어핀(레이싱라인 s ≈ 24 m)에서는 빠져나갈 방향이 시야 끝(-85°)에 있고, 핸들을 끝까지 꺾어도 회전이 모자라 바깥 벽에 닿는다.
명령 지연 30 ms와 라이다 위치(뒷바퀴 축보다 0.27 m 앞) 때문에 더 나빠진다.

### 1.2 v3: FTG + 원호 검사 (`FTGArcDriver`)
매 주기(50 Hz)마다:
1. **방향**: v2(`FTGDriver`, 스무딩 끔)가 원하는 조향각 `target`을 낸다.
2. **원호 검사**: 조향 후보 31개(-24° ~ +24°)마다 자전거 모델로 차가 그릴 원호를 5 m 앞까지 그린다(곡률 = tan(조향각) / 축간거리 0.33 m).
   원호를 0.1 m 간격으로 따라가며 **차 앞부분 위치**에서 라이다 점까지의 거리를 재고, 차 반폭 + 여유(0.22 m)보다 가까워지기 전까지의 거리를 `free`로 둔다.
   거리는 라이다 점을 4 cm 지역 격자에 찍고 거리 변환(`scipy.ndimage.distance_transform_edt`)으로 한 번에 구한 지도에서 읽는다(1회 계산 약 1 ms).
3. **선택**: 점수 = min(`free`, 3 m) − 1.5 × |조향 − `target`| − 0.3 × |조향 − 직전 조향|. 점수가 가장 높은 후보를 고른다.
4. **속도**: min(√(2 × 5 m/s² × (`free` − 0.35 m)), √(8 m/s² ÷ 곡률)), 0.3~7 m/s로 자른다.
   → 막히기 전에 설 수 있는 속도와, 그 곡선에서 미끄러지지 않는 속도 중 작은 값.

왜 둘을 합쳤나
- 원호 검사만 쓰면(가장 멀리 갈 수 있는 원호) 넓은 구간에서 U턴해 역주행했다. 방향 감각은 FTG가 낫다.
- FTG만 쓰면 회전 가능성과 감속을 모른다. 원호 검사가 이를 보완한다.

## 2. 실행 방법 (도커 `misys:forza_full` 컨테이너 안, 프롬프트가 `misys@...`)

### 2.1 준비 (처음 한 번)
```bash
cd ~
git clone https://github.com/shapple0120-crypto/car.git                  # 이 저장소
git clone https://github.com/delboy11/forza-obstacle-avoider.git        # harness.py가 있는 팀원 저장소
git clone https://github.com/delboy11/forza-lidar-corridor-mpc.git      # run_avoid.sh(ROS 실행)가 있는 팀원 저장소
pip install --user -r ~/forza-obstacle-avoider/requirements.txt

# 내 주행기 파일 2개를 팀원 저장소 drivers 폴더에 바로가기(심볼릭 링크)로 연결한다.
# v3가 같은 폴더의 ftg_driver.py를 불러 쓰므로 두 파일 모두 연결해야 한다.
for f in ftg_driver.py ftg_arc_driver.py; do
  ln -sf ~/car/drivers/$f ~/forza-obstacle-avoider/drivers/$f
  ln -sf ~/car/drivers/$f ~/forza-lidar-corridor-mpc/drivers/$f
done
```
- berlin 맵이 `~/forza_ws/race_stack/stack_master/maps/berlin/`에 있어야 한다(`berlin.png`, `berlin.yaml`, `global_waypoints.json`).
  없으면 도커 **밖**(WSL)에서 `docker cp <berlin 폴더 경로> <컨테이너 이름>:/home/misys/forza_ws/race_stack/stack_master/maps/`로 넣는다.
- 저장소를 갱신하면(`cd ~/car && git pull`) 바로가기 덕분에 팀원 저장소 쪽에도 바로 반영된다.

### 2.2 harness로 빠르게 평가 (ROS 없이, 여러 seed 병렬)
```bash
source /opt/ros/humble/setup.bash && source ~/forza_ws/race_stack/install/setup.bash
cd ~/forza-obstacle-avoider
python3 harness.py drivers.ftg_arc_driver:FTGArcDriver --map berlin --seeds 0-9 --laps 10 --obstacles 3 --jobs 4 --out arc_o3.json
```
- v2와 비교하려면 `drivers.ftg_arc_driver:FTGArcDriver` 대신 `drivers.ftg_driver:FTGDriver`.
- `--obstacles 1 / 3 / 5`: 최종 평가 조건. `--seeds 0-9`: seed 10개. `--laps 10`: 평가와 같은 10바퀴.
- 노이즈 테스트: `--scan-noise 0.02`(라이다 거리 잡음 표준편차 2 cm), `--scan-drop 0.01`(빔 1% 누락)
- 파라미터 바꾸기: `--kw '{"w_dir": 1.0, "v_max": 6.0}'`
- 출력 한 줄 요약: `laps`, `collisions`(= `obstacle_hits` + `wall_hits`), `collisions_per_lap`, `stalls`(6초 이상 정지), `clean_lap_time_mean`(무충돌 랩 평균), `act_ms_mean`(1회 계산 시간)
- `--out` 파일에는 충돌마다 위치(`s`), 속도(`v`), 종류(`kind`)가 남는다.
- 주의: harness는 충돌하면 차를 앞에 다시 놓고 계속 달린다. 최종 평가(충돌하면 run 종료)와 규칙이 달라서 ROS 실시뮬 숫자와 섞어 비교하지 않는다.

### 2.3 ROS 실시뮬로 보기 (RViz + race_manager 채점)
```bash
source /opt/ros/humble/setup.bash && source ~/forza_ws/race_stack/install/setup.bash
cd ~/forza-lidar-corridor-mpc
./run_avoid.sh drivers.ftg_arc_driver:FTGArcDriver '{}' 3 7 berlin      # 장애물 3개, seed 7, berlin
```
- 인자 순서: `<module:Class> '<json kwargs>' [장애물 수] [seed(-1이면 무작위)] [맵]`. 종료는 Ctrl-C.
- 결과는 `~/forza_ws/race_stack/race_result/`의 json. GT 미구독 확인: `ros2 node info /avoid_driver`

## 3. 결과

### 3.1 harness (팀 공통 평가 도구, 2026-11-23, 도커 PC 8코어 `--jobs 7`)
`python3 harness.py drivers.<파일>:<클래스> --map berlin --seeds 0-9 --laps 10 --obstacles <1|3|5> --jobs 7`
(harness는 충돌해도 차를 앞에 다시 놓고 계속 달린다. 10바퀴 연속 무충돌 확률은 바퀴당 충돌 λ로 어림한 e^(−10λ))

| 장애물 | 주행기 | 완주 랩 | 충돌 (장애물 / 벽) | 충돌/랩 | 무충돌 랩 평균 | 시간 초과 판 | act ms 평균 / p99 최대 | 10바퀴 무충돌 확률(근사) |
|---|---|---|---|---|---|---|---|---|
| 1 | v2 | 100 | 78 (0 / 78) | 0.78 | 12.26 s | 0 | 0.65 / 7.4 | ≈ 0% |
| 1 | **v3** | 88 | 16 (5 / 11) | **0.18** | **10.08 s** | 2 | 5.6 / 17.3 | ≈ 16% |
| 3 | v2 | 100 | 81 (7 / 74) | 0.81 | 13.26 s | 0 | 0.86 / 9.6 | ≈ 0% |
| 3 | **v3** | 91 | 30 (6 / 24) | **0.33** | **11.43 s** | 1 | 4.6 / 15.2 | ≈ 4% |
| 5 | v2 | 94 | 218 (75 / 143) | 2.32 | 16.29 s | 2 | 0.75 / 9.8 | ≈ 0% |
| 5 | **v3** | 90 | 50 (20 / 30) | **0.56** | **11.47 s** | 2 | 4.9 / 15.5 | ≈ 0.4% |

- v3는 모든 조건에서 충돌 2.5~4배 감소, 무충돌 랩 1~5 s 단축. 장애물 3 → 5개에서도 랩타임이 거의 그대로(11.43 → 11.47 s).
- **v3 벽 충돌의 위치(장애물 3개)**: 24번 중 22번이 레이싱라인 s = 23.7~24.8 m(두 번째 헤어핀 입구) 한 곳. 이 구간을 원리적으로(하드코딩 없이) 해결하면 충돌/랩 0.33 → 약 0.09로 줄 수 있다 → 다음 개선 대상.
- 시간 초과 판: 막힌 곳에서 0.3 m/s로 기어가다 시뮬 시간 400 s 제한에 걸린 판.
- act ms는 7판을 동시에 돌린 값이라 단독 실행보다 크다. ROS 제어 주기 20 ms 안이지만 ROS 실시뮬에서 따로 확인할 것.

### 3.2 간이 시뮬 (harness 조건 흉내: 명령 지연 30 ms, 라이다 0.27 m 앞, 라이다 중심 0.33 × 0.20 m 충돌 박스)
공개 f1tenth_gym 물리 + berlin 맵 + 레이싱라인 근처 0.3 m 정사각형 장애물(배치 규칙은 race_manager와 다름). seed 10개(v2 장애물 3개는 6개), 10바퀴, **충돌하면 종료**(평가 규칙과 같음).

| 장애물 | v2: 10바퀴 완주 | v2: 평균 연속 랩 | v3: 10바퀴 완주 | v3: 평균 연속 랩 | v3 best lap |
|---|---|---|---|---|---|
| 1 | 0/10 | 0.7 | 6/10 | 7.4 | 9.54 s |
| 3 | 0/6 | 0.5 | 4/10 | 5.9 | 9.56 s |
| 5 | 0/10 | 2.4 | 4/10 | 4.1 | 9.75 s |

- v3는 v2보다 확실히 낫지만, 아직 절반 이상은 10바퀴 전에 끝난다.
- v3의 남은 실패: 첫 헤어핀(s ≈ 23~24 m)에서 0.3 m/s로 벽에 닿거나 멈춤, 장애물 근처에서 2~5 m/s로 장애물 접촉.
- 원호 검사를 격자 거리 변환으로 바꿔 1회 계산이 약 8 ms → 1 ms로 줄었다(결과는 같거나 조금 나음). 8 ms 버전은 도커 PC에서 harness 한 판에 10분 가까이 걸렸다.
- 숫자는 harness·ROS 결과와 직접 비교할 수 없다. 도커에서 harness로 다시 측정한다(3.1).

### 3.3 v2 개선 시도 기록 (모두 효과 없음 → 구조를 바꾼 v3로 감)
지연·라이다 위치를 넣은 간이 시뮬에서 v2의 파라미터를 바꿔 봤다: 조향 스무딩 0, 최고 속도 4.5, 차 폭 여유 0.30, `gap_threshold` 2.0, 최저 속도 1.0~1.2, 시야 100°, 조향 배율 1.5~2.0, 옆 가드, 물리 기반 속도.
모두 같거나 나빴다. 지연이 없는 간이 시뮬에서도 `max_speed` 7은 빨라지지만 충돌이 급증했다(10바퀴 완주 8/16 → 2/16).

### 3.4 결론
- 반응형은 매 순간 보이는 것에만 반응하므로, 헤어핀처럼 "미리 줄이고 미리 붙어야 하는" 구간에 약하다. 원호 검사로 회전 가능성을 따지면 크게 나아지지만 10바퀴 연속 완주를 보장하지 못한다.
- 팀 최종 주행기(LiDAR CorridorMPC)는 지도 위 자기 위치와 장애물 기억으로 몇 m 앞을 계획한다. harness 장애물 4~5개 1800랩에서 충돌 8번(0.0044회/랩), 평균 랩 약 8.6 s(팀원 README).

## 4. 파일
| 파일 | 내용 |
|---|---|
| `drivers/ftg_arc_driver.py` | `FTGArcDriver` (v3, FTG + 원호 검사) |
| `drivers/ftg_driver.py` | `FTGDriver` (v2, Follow-the-Gap + Disparity Extender) |
| `drivers/__init__.py` | 패키지 표시용 빈 파일 |
| `tools/mini_sim.py` | 간이 시뮬레이터 (ROS·race_manager 없이 동작 확인용, harness 조건 흉내) |

### v2 변경점 (v1 리뷰 반영)
| 문제 | 수정 |
|---|---|
| `avoid_node.py`가 GT 토픽을 구독함 | `gt_free = True` |
| 3칸 평균 후 모서리 검사 → 단차가 약 1 m보다 작은 모서리를 놓침 | 모서리 검사를 원본 거리로 |
| 시야각 270°(4.712 rad) | 시뮬레이터 값 4.7 rad |
| 벽에 붙으면 안전 거품이 양쪽 약 67°씩 막음 | 최대 30°로 제한 |
| 정면이 막혀도 최저 1.8 m/s | 0.6 m 안쪽이면 0.5 m/s |
