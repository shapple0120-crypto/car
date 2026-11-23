import numpy as np  # 숫자 배열(라이다 거리 목록)을 빠르게 계산하기 위한 도구


class FTGDriver:
    """
    Follow-the-Gap + Disparity Extender 반응형(Reactive) 주행기
    (기준선 비교용 Baseline 드라이버)
    """

    def __init__(
        self,
        fov_rad: float = np.deg2rad(270.0),       # 라이다 전체 시야각 (F1TENTH 표준 270도)
        front_angle_deg: float = 85.0,            # 정면 기준 좌우 몇 도까지만 볼지 (85도 -> 총 170도)
        max_lidar_dist: float = 10.0,             # 너무 먼 거리는 10m로 잘라서 노이즈 영향 줄이기
        car_half_width: float = 0.25,             # 차 폭의 절반 + 안전 여유 (m). 0.15~0.30 사이로 튜닝
        disparity_threshold: float = 0.35,        # 옆 레이저와 거리 차이가 이 값(m) 넘으면 "모서리"로 판단
        bubble_radius: float = 0.35,              # 가장 가까운 장애물 주변을 막아둘 안전 거품 반경 (m)
        gap_threshold: float = 1.2,               # 이 거리(m)보다 멀리 뚫려 있어야 "지나갈 빈틈(Gap)"으로 인정
        max_steer: float = 0.4189,                # 최대 조향각 (라디안, 약 24도. F1TENTH 표준 한계)
        max_speed: float = 5.5,                   # 직선 최고 속도 (m/s). 처음엔 4.0~5.5로 안전하게 시작
        min_speed: float = 1.8,                   # 급커브/장애물 근접 시 최저 속도 (m/s)
        steer_smooth: float = 0.35,               # 핸들이 덜덜 떨리지 않게 이전 각도와 섞는 비율 (0~1)
    ):
        # 아래 줄들은 위에서 받은 설정값들을 이 주행기 객체 안에 저장해 두는 작업
        self.fov_rad = fov_rad
        self.front_angle_rad = np.deg2rad(front_angle_deg)
        self.max_lidar_dist = max_lidar_dist
        self.car_half_width = car_half_width
        self.disparity_threshold = disparity_threshold
        self.bubble_radius = bubble_radius
        self.gap_threshold = gap_threshold
        self.max_steer = max_steer
        self.max_speed = max_speed
        self.min_speed = min_speed
        self.steer_smooth = steer_smooth

        # 직전 프레임의 조향각을 기억해 두기 위한 변수 (부드러운 핸들링용)
        self.prev_steer = 0.0

    def reset(self):
        """새 랩이나 새 시드가 시작될 때 내부 기억을 초기화하는 함수"""
        self.prev_steer = 0.0  # 이전 조향각 기억을 0(직진)으로 리셋

    def _preprocess_scan(self, scan: np.ndarray) -> np.ndarray:
        """1단계: 라이다 이상값(NaN, 무한대) 정리 및 부드럽게 다듬기"""
        # 원본 배열을 건드리지 않도록 복사하고 실수형(float64)으로 변환
        ranges = np.array(scan, dtype=np.float64, copy=True)
        # 측정 실패(NaN)는 0m(위험), 무한대(inf)는 최대 거리로 바꿔서 에러 방지
        ranges = np.nan_to_num(
            ranges, nan=0.0, posinf=self.max_lidar_dist, neginf=0.0
        )
        # 0m 미만이나 max_lidar_dist 초과 값을 지정 범위 안으로 자르기
        ranges = np.clip(ranges, 0.0, self.max_lidar_dist)
        # 3칸짜리 이동 평균(주변 3개 값의 평균)으로 튀는 라이다 노이즈 완화
        kernel = np.ones(3) / 3.0
        ranges = np.convolve(ranges, kernel, mode="same")
        return ranges

    def _apply_disparity_extender(
        self, ranges: np.ndarray, angle_inc: float
    ) -> np.ndarray:
        """2단계: Disparity Extender - 장애물 모서리를 차 폭만큼 옆으로 부풀리기"""
        extended = ranges.copy()  # 원본은 비교용으로 두고, 수정할 복사본 생성
        n = len(ranges)           # 전방 시야 안의 레이저 개수

        # 인접한 두 레이저(i번째와 i+1번째)의 거리 차이를 차례대로 검사
        for i in range(n - 1):
            d1 = ranges[i]        # 오른쪽 레이저 거리
            d2 = ranges[i + 1]    # 왼쪽 레이저 거리
            diff = d2 - d1        # 거리 변화량 계산

            # 거리 차이가 기준(disparity_threshold)보다 작으면 평범한 벽/바닥이므로 통과
            if abs(diff) < self.disparity_threshold:
                continue

            # 두 점 중 더 가까운 쪽이 실제 장애물(또는 코너)의 모서리임
            closer_dist = min(d1, d2)
            # 거리가 0에 가까우면 나눗셈 에러 방지를 위해 최소 0.05m로 보정
            closer_dist = max(closer_dist, 0.05)

            # "거리 closer_dist에서 차 반폭만큼 비키려면 레이저를 몇 칸이나 덮어야 하는가?" 계산
            # 부채꼴 호의 길이 공식: 각도 = 호의 길이(차 반폭) / 반지름(거리)
            needed_angle = self.car_half_width / closer_dist
            pad_beams = int(np.ceil(needed_angle / angle_inc))  # 필요한 레이저 칸 수(올림)

            if diff > 0:
                # 오른쪽(i)이 가깝고 왼쪽(i+1)이 멀다 -> 왼쪽 방향으로 모서리를 부풀림
                end_idx = min(n, i + 1 + pad_beams)
                extended[i + 1 : end_idx] = np.minimum(
                    extended[i + 1 : end_idx], closer_dist
                )
            else:
                # 왼쪽(i+1)이 가깝고 오른쪽(i)이 멀다 -> 오른쪽 방향으로 모서리를 부풀림
                start_idx = max(0, i - pad_beams + 1)
                extended[start_idx : i + 1] = np.minimum(
                    extended[start_idx : i + 1], closer_dist
                )

        return extended

    def _apply_safety_bubble(
        self, ranges: np.ndarray, angle_inc: float
    ) -> np.ndarray:
        """3단계: 가장 가까운 위험 물체 주변을 0m(진입 금지)로 막기"""
        out = ranges.copy()
        closest_idx = int(np.argmin(out))          # 현재 제일 가까운 점의 위치(인덱스) 찾기
        closest_dist = max(out[closest_idx], 0.05) # 그 점까지의 거리

        # 가장 가까운 점 양옆으로 bubble_radius만큼의 레이저 칸 수 계산
        bubble_beams = int(np.ceil((self.bubble_radius / closest_dist) / angle_inc))
        start_idx = max(0, closest_idx - bubble_beams)
        end_idx = min(len(out), closest_idx + bubble_beams + 1)
        out[start_idx:end_idx] = 0.0  # 이 구간은 절대 가지 못하게 거리 0으로 설정
        return out

    def _find_best_gap(self, ranges: np.ndarray) -> tuple[int, int]:
        """4단계: gap_threshold보다 멀리 뚫려 있는 구간 중 '가장 넓은 빈틈' 찾기"""
        # 각 레이저가 기준 거리보다 멀면 True(통과 가능), 아니면 False(막힘)
        free_mask = ranges > self.gap_threshold

        best_start, best_end = 0, 0   # 지금까지 찾은 가장 넓은 빈틈의 시작과 끝
        cur_start = None              # 현재 측정 중인 빈틈의 시작점

        for i, is_free in enumerate(free_mask):
            if is_free and cur_start is None:
                cur_start = i         # 새 빈틈 시작!
            elif not is_free and cur_start is not None:
                # 빈틈이 끝났으므로, 기존 최고 기록보다 넓으면 교체
                if (i - cur_start) > (best_end - best_start):
                    best_start, best_end = cur_start, i
                cur_start = None      # 다음 빈틈을 찾기 위해 초기화

        # 배열 맨 끝까지 빈틈이 이어진 경우 마무리 확인
        if cur_start is not None:
            if (len(ranges) - cur_start) > (best_end - best_start):
                best_start, best_end = cur_start, len(ranges)

        # 만약 주변이 너무 좁아서 기준 넘는 빈틈이 하나도 없다면, 그나마 제일 먼 점 선택
        if best_end == best_start:
            farthest = int(np.argmax(ranges))
            return farthest, farthest + 1

        return best_start, best_end

    def _choose_target_index(
        self, ranges: np.ndarray, start: int, end: int
    ) -> int:
        """5단계: 찾은 빈틈 안에서 실제로 겨냥할 목표 각도(인덱스) 고르기"""
        gap_slice = ranges[start:end]          # 선택된 빈틈 구간만 잘라내기
        max_dist = np.max(gap_slice)           # 그 빈틈 안에서 가장 먼 거리
        # 최대 거리의 92% 이상으로 멀리 뚫린 지점들의 '평균 위치'를 조준
        # (가장 먼 점 1개만 찍으면 벽 끝으로 쏠려서 흔들리는 현상을 방지)
        deep_indices = np.where(gap_slice >= max_dist * 0.92)[0]
        target_offset = int(np.mean(deep_indices))
        return start + target_offset

    def act(self, scan, x, y, yaw, v, yaw_rate, t):
        """
        매 순간(0.02초마다) 호출되는 핵심 운전 함수
        입력: 라이다(scan), 위치/속도 정보(FTG는 안 쓰고 scan만 사용 -> GT 미사용!)
        출력: (steer: 조향각, speed: 목표 속도)
        """
        # 1. 라이다 전체 전처리
        full_ranges = self._preprocess_scan(scan)
        n_total = len(full_ranges)
        # 레이저 1칸당 각도 간격 (예: 270도 / 1079칸 = 약 0.25도)
        angle_inc = self.fov_rad / max(n_total - 1, 1)

        # 2. 전체 시야 중 앞쪽(-front_angle ~ +front_angle)만 잘라내기
        mid = n_total // 2                                    # 정면(0도) 인덱스
        half_span = int(self.front_angle_rad / angle_inc)     # 정면 좌우로 몇 칸 볼지
        left_cut = max(0, mid - half_span)                    # 잘라낼 시작 인덱스
        right_cut = min(n_total, mid + half_span + 1)         # 잘라낼 끝 인덱스
        front_ranges = full_ranges[left_cut:right_cut]        # 전방 라이다만 추출

        # 3. Disparity Extender(모서리 부풀리기) + 안전 거품 적용
        safe_ranges = self._apply_disparity_extender(front_ranges, angle_inc)
        safe_ranges = self._apply_safety_bubble(safe_ranges, angle_inc)

        # 4. 가장 넓은 빈틈을 찾고, 그 안에서 목표 지점 고르기
        gap_start, gap_end = self._find_best_gap(safe_ranges)
        target_idx = self._choose_target_index(safe_ranges, gap_start, gap_end)

        # 5. 목표 인덱스를 실제 핸들 각도(라디안)로 변환
        front_mid = len(safe_ranges) // 2                     # 전방 배열에서의 정면 위치
        raw_steer = (target_idx - front_mid) * angle_inc      # 정면에서 몇 라디안 떨어졌는지 계산
        raw_steer = float(np.clip(raw_steer, -self.max_steer, self.max_steer))

        # 핸들이 갑자기 홱 꺾이지 않도록 이전 각도와 부드럽게 섞기 (로우패스 필터)
        steer = self.steer_smooth * self.prev_steer + (1.0 - self.steer_smooth) * raw_steer
        self.prev_steer = steer

        # 6. 속도 결정: 핸들을 많이 꺾었거나, 정면이 가까우면 감속
        steer_ratio = abs(steer) / self.max_steer             # 핸들을 최대로 꺾은 비율 (0.0 ~ 1.0)
        # 정면 10도 정도 범위의 최소 거리 확인 (앞에 뭐가 가까이 있으면 감속)
        center_margin = max(1, int(np.deg2rad(10.0) / angle_inc))
        front_min_dist = float(
            np.min(front_ranges[front_mid - center_margin : front_mid + center_margin + 1])
        )

        # 핸들 각도에 따른 속도 계산 (직선이면 max_speed, 급커브면 min_speed)
        speed = self.max_speed - (self.max_speed - self.min_speed) * (steer_ratio ** 0.8)

        # 바로 앞이 2.5m 이내로 막혀 있으면 추가 감속
        if front_min_dist < 2.5:
            dist_factor = np.clip((front_min_dist - 0.5) / 2.0, 0.0, 1.0)
            speed = min(speed, self.min_speed + (self.max_speed - self.min_speed) * dist_factor)

        speed = float(np.clip(speed, self.min_speed, self.max_speed))
        return steer, speed