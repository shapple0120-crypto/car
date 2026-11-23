"""
Follow-the-Gap + Disparity Extender 반응형(Reactive) 주행기 (비교 기준선 Baseline 드라이버)

- 팀 공통 인터페이스: act(scan[1080], x, y, yaw, v, yaw_rate, t) -> (steer, speed), reset()
  -> 팀원 저장소의 avoid_node.py(ROS 노드)와 harness.py(오프라인 평가)에 그대로 끼워 쓸 수 있다.
- 라이다(scan)만 쓴다. x, y, yaw, v, yaw_rate는 받기만 하고 쓰지 않는다 (주행 중 GT 미사용).
- 지도, 레이싱라인, 맵 좌표를 전혀 쓰지 않으므로 구간별 하드코딩이 없다.

v2 변경점 (리뷰 반영)
1. gt_free = True: avoid_node.py가 GT 토픽(/car_state/*_GT)을 구독하지 않게 한다.
2. 모서리(disparity) 검사를 스무딩 전 원본 거리로 한다. 3칸 평균을 먼저 하면 큰 단차가 작은 단차 3개로
   쪼개져서, 단차가 약 1 m보다 작은 장애물 모서리를 놓쳤다.
3. 라이다 시야각을 시뮬레이터 값(4.7 rad, 1080빔)에 맞췄다.
4. 안전 거품 각도에 상한(기본 30도)을 둬서, 벽에 붙었을 때 시야 대부분이 지워지지 않게 했다.
5. 정면이 아주 가까우면 min_speed 아래(stop_speed)까지 감속할 수 있게 했다.
"""
import numpy as np  # 숫자 배열(라이다 거리 목록)을 빠르게 계산하기 위한 도구


class FTGDriver:
    """Follow-the-Gap + Disparity Extender 주행기"""

    # avoid_node.py는 주행기의 gt_free 값을 보고 GT 토픽 구독 여부를 정한다.
    # True이면 /scan과 /initialpose만 구독한다 (ros2 node info /avoid_driver로 확인 가능).
    gt_free = True

    def __init__(
        self,
        fov_rad: float = 4.7,                     # 라이다 전체 시야각 (시뮬레이터: 4.7 rad, 1080빔)
        front_angle_deg: float = 85.0,            # 정면 기준 좌우 몇 도까지만 볼지 (85도 -> 총 170도)
        max_lidar_dist: float = 10.0,             # 너무 먼 거리는 10 m로 잘라서 노이즈 영향 줄이기
        car_half_width: float = 0.25,             # 차 폭의 절반 + 안전 여유 (m). 0.15~0.30 사이로 튜닝
        disparity_threshold: float = 0.35,        # 옆 레이저와 거리 차이가 이 값(m) 넘으면 "모서리"로 판단
        bubble_radius: float = 0.35,              # 가장 가까운 점 주변을 막아둘 안전 거품 반경 (m)
        bubble_max_deg: float = 30.0,             # 안전 거품이 한쪽으로 막을 수 있는 최대 각도 (도)
        gap_threshold: float = 1.2,               # 이 거리(m)보다 멀리 뚫려 있어야 "지나갈 빈틈(Gap)"으로 인정
        max_steer: float = 0.4189,                # 최대 조향각 (라디안, 약 24도)
        max_speed: float = 5.5,                   # 직선 최고 속도 (m/s). 처음엔 4.0~5.5로 안전하게 시작
        min_speed: float = 1.8,                   # 급커브/장애물 근접 시 평소 최저 속도 (m/s)
        stop_speed: float = 0.5,                  # 정면이 아주 가까울 때 내려갈 수 있는 최저 속도 (m/s)
        stop_dist: float = 0.6,                   # 정면 최소 거리가 이 값(m)보다 가까우면 stop_speed까지 감속
        steer_smooth: float = 0.35,               # 핸들 떨림 방지: 이전 각도와 섞는 비율 (0~1)
    ):
        # 받은 설정값들을 이 주행기 객체 안에 저장한다
        self.fov_rad = fov_rad
        self.front_angle_rad = np.deg2rad(front_angle_deg)   # 도 -> 라디안 변환
        self.max_lidar_dist = max_lidar_dist
        self.car_half_width = car_half_width
        self.disparity_threshold = disparity_threshold
        self.bubble_radius = bubble_radius
        self.bubble_max_rad = np.deg2rad(bubble_max_deg)      # 도 -> 라디안 변환
        self.gap_threshold = gap_threshold
        self.max_steer = max_steer
        self.max_speed = max_speed
        self.min_speed = min_speed
        self.stop_speed = stop_speed
        self.stop_dist = stop_dist
        self.steer_smooth = steer_smooth

        # 직전 프레임의 조향각을 기억해 두는 변수 (부드러운 핸들링용)
        self.prev_steer = 0.0

    def reset(self):
        """새 에피소드가 시작되거나 차를 다시 놓았을 때 내부 기억을 초기화한다"""
        self.prev_steer = 0.0  # 이전 조향각 기억을 0(직진)으로 리셋

    def _clean_scan(self, scan) -> np.ndarray:
        """1-a단계: 라이다 이상값(NaN, 무한대)만 정리한 '원본' 거리"""
        ranges = np.array(scan, dtype=np.float64, copy=True)  # 원본을 건드리지 않게 복사 + 실수형 변환
        # 측정 실패(NaN)는 0 m(위험), 무한대(inf)는 최대 거리로 바꿔서 계산 에러를 막는다
        ranges = np.nan_to_num(ranges, nan=0.0, posinf=self.max_lidar_dist, neginf=0.0)
        # 0 m 미만이나 최대 거리 초과 값을 범위 안으로 자른다
        return np.clip(ranges, 0.0, self.max_lidar_dist)

    @staticmethod
    def _smooth(ranges: np.ndarray) -> np.ndarray:
        """1-b단계: 3칸 이동 평균으로 튀는 노이즈를 줄인 '부드러운' 거리 (빈틈 찾기·속도 계산용)"""
        kernel = np.ones(3) / 3.0                                # 주변 3개 값을 똑같은 비율로 섞는 필터
        out = np.convolve(ranges, kernel, mode="same")           # 각 칸 = 자기 + 양옆 평균
        out[0], out[-1] = ranges[0], ranges[-1]                  # 양 끝은 옆 칸이 하나뿐이라 원래 값 유지
        return out

    def _apply_disparity_extender(self, ranges: np.ndarray, angle_inc: float) -> np.ndarray:
        """2단계: Disparity Extender - 장애물 모서리를 차 반폭만큼 옆으로 부풀린다 (원본 거리로 검사)"""
        extended = ranges.copy()  # 원본은 비교용으로 두고, 수정할 복사본을 만든다
        n = len(ranges)           # 전방 시야 안의 레이저 개수

        diffs = np.diff(ranges)                                      # 인접한 두 레이저의 거리 차이 (i+1 - i)
        edges = np.nonzero(np.abs(diffs) >= self.disparity_threshold)[0]  # 차이가 큰 곳 = 모서리 후보 위치

        for i in edges:                                   # 모서리 후보만 돌아서 계산을 줄인다
            d1, d2 = ranges[i], ranges[i + 1]             # 오른쪽(i), 왼쪽(i+1) 레이저 거리
            closer_dist = max(min(d1, d2), 0.05)          # 더 가까운 쪽이 모서리. 0으로 나누기 방지로 최소 0.05 m

            # "거리 closer_dist에서 차 반폭만큼 비키려면 몇 칸을 덮어야 하나?" (각도 = 호의 길이 / 반지름)
            pad_beams = int(np.ceil((self.car_half_width / closer_dist) / angle_inc))

            if d2 > d1:
                # 오른쪽이 가깝고 왼쪽이 멀다 -> 왼쪽(먼 쪽) 칸들을 가까운 거리로 덮는다
                end_idx = min(n, i + 1 + pad_beams)
                extended[i + 1:end_idx] = np.minimum(extended[i + 1:end_idx], closer_dist)
            else:
                # 왼쪽이 가깝고 오른쪽이 멀다 -> 오른쪽(먼 쪽) 칸들을 가까운 거리로 덮는다
                start_idx = max(0, i - pad_beams + 1)
                extended[start_idx:i + 1] = np.minimum(extended[start_idx:i + 1], closer_dist)

        return extended

    def _apply_safety_bubble(self, ranges: np.ndarray, angle_inc: float) -> np.ndarray:
        """3단계: 가장 가까운 점 주변을 0 m(진입 금지)로 막는다. 막는 각도에는 상한을 둔다"""
        out = ranges.copy()
        closest_idx = int(np.argmin(out))                # 제일 가까운 점의 위치(인덱스)
        closest_dist = max(out[closest_idx], 0.05)       # 그 점까지의 거리 (0 나누기 방지)

        # 반경 / 거리 = 막을 각도. 벽에 붙으면 각도가 너무 커지므로 bubble_max_rad로 제한한다
        bubble_angle = min(self.bubble_radius / closest_dist, self.bubble_max_rad)
        bubble_beams = int(np.ceil(bubble_angle / angle_inc))     # 각도 -> 레이저 칸 수
        start_idx = max(0, closest_idx - bubble_beams)
        end_idx = min(len(out), closest_idx + bubble_beams + 1)
        out[start_idx:end_idx] = 0.0                     # 이 구간은 갈 수 없게 거리 0으로 설정
        return out

    def _find_best_gap(self, ranges: np.ndarray) -> tuple:
        """4단계: gap_threshold보다 멀리 뚫린 구간 중 '가장 넓은 빈틈'을 찾는다 -> (시작, 끝) 인덱스"""
        free = ranges > self.gap_threshold               # 기준보다 멀면 True(통과 가능)
        best_start, best_end = 0, 0                      # 지금까지 찾은 가장 넓은 빈틈
        cur_start = None                                 # 지금 재고 있는 빈틈의 시작점

        for i, is_free in enumerate(free):
            if is_free and cur_start is None:
                cur_start = i                            # 새 빈틈 시작
            elif not is_free and cur_start is not None:
                if (i - cur_start) > (best_end - best_start):   # 기존 최고보다 넓으면 교체
                    best_start, best_end = cur_start, i
                cur_start = None                         # 다음 빈틈을 위해 초기화

        # 배열 끝까지 빈틈이 이어진 경우 마무리
        if cur_start is not None and (len(ranges) - cur_start) > (best_end - best_start):
            best_start, best_end = cur_start, len(ranges)

        # 기준을 넘는 빈틈이 하나도 없으면, 그나마 제일 먼 점 하나를 고른다
        if best_end == best_start:
            farthest = int(np.argmax(ranges))
            return farthest, farthest + 1
        return best_start, best_end

    @staticmethod
    def _choose_target_index(ranges: np.ndarray, start: int, end: int) -> int:
        """5단계: 빈틈 안에서 겨냥할 목표 인덱스를 고른다"""
        gap_slice = ranges[start:end]                    # 선택된 빈틈 구간만 잘라낸다
        max_dist = np.max(gap_slice)                     # 그 안에서 가장 먼 거리
        # 최대 거리의 92% 이상인 지점들의 '평균 위치'를 조준한다 (벽 끝으로 쏠려 흔들리는 것 방지)
        deep = np.nonzero(gap_slice >= max_dist * 0.92)[0]
        return start + int(np.mean(deep))

    def act(self, scan, x, y, yaw, v, yaw_rate, t):
        """
        매 제어 주기(50 Hz = 0.02초마다) 호출되는 운전 함수
        입력: scan(라이다 1080개 거리). x, y, yaw, v, yaw_rate, t는 쓰지 않는다 (GT 미사용)
        출력: (steer: 조향각 rad, 왼쪽이 +), (speed: 목표 속도 m/s)
        """
        # 1. 라이다 정리: 원본(raw)과 부드러운 값(smooth)을 따로 만든다
        raw = self._clean_scan(scan)
        smooth = self._smooth(raw)
        n_total = len(raw)
        angle_inc = self.fov_rad / max(n_total - 1, 1)        # 레이저 1칸당 각도 (4.7 / 1079 ≈ 0.25도)

        # 2. 전체 시야 중 앞쪽(-front_angle ~ +front_angle)만 잘라낸다. 인덱스 0이 오른쪽 끝, 가운데가 정면
        mid = n_total // 2                                     # 정면(0도) 인덱스
        half_span = int(self.front_angle_rad / angle_inc)      # 정면 좌우로 몇 칸 볼지
        lo, hi = max(0, mid - half_span), min(n_total, mid + half_span + 1)
        front_raw = raw[lo:hi]                                 # 전방 원본
        front_smooth = smooth[lo:hi]                           # 전방 부드러운 값

        # 3. 모서리 부풀리기는 원본으로 검사하고, 부드러운 값과 비교해 더 가까운(보수적인) 쪽을 쓴다
        safe = self._apply_disparity_extender(front_raw, angle_inc)
        safe = np.minimum(safe, front_smooth)
        safe = self._apply_safety_bubble(safe, angle_inc)

        # 4. 가장 넓은 빈틈을 찾고, 그 안에서 목표 지점을 고른다
        gap_start, gap_end = self._find_best_gap(safe)
        target_idx = self._choose_target_index(safe, gap_start, gap_end)

        # 5. 목표 인덱스 -> 조향각. 정면보다 왼쪽(인덱스 큼)이면 +(왼쪽으로 꺾기)
        front_mid = mid - lo                                   # 전방 배열 안에서의 정면 위치
        raw_steer = float(np.clip((target_idx - front_mid) * angle_inc, -self.max_steer, self.max_steer))

        # 핸들이 갑자기 홱 꺾이지 않게 이전 각도와 섞는다 (로우패스 필터)
        steer = self.steer_smooth * self.prev_steer + (1.0 - self.steer_smooth) * raw_steer
        self.prev_steer = steer

        # 6. 속도: 핸들을 많이 꺾을수록, 정면이 가까울수록 느리게
        steer_ratio = abs(steer) / self.max_steer             # 최대 조향 대비 비율 (0~1)
        center = max(1, int(np.deg2rad(10.0) / angle_inc))    # 정면 ±10도에 해당하는 칸 수
        front_min = float(np.min(front_smooth[front_mid - center:front_mid + center + 1]))  # 정면 최소 거리

        speed = self.max_speed - (self.max_speed - self.min_speed) * (steer_ratio ** 0.8)  # 직선 빠르게, 커브 느리게
        if front_min < 2.5:                                    # 앞이 2.5 m 안쪽으로 막혀 있으면 추가 감속
            k = np.clip((front_min - 0.5) / 2.0, 0.0, 1.0)
            speed = min(speed, self.min_speed + (self.max_speed - self.min_speed) * k)
        speed = float(np.clip(speed, self.min_speed, self.max_speed))

        if front_min < self.stop_dist:                         # 바로 앞이 아주 가까우면 min_speed 아래로도 감속
            speed = self.stop_speed

        return steer, speed
