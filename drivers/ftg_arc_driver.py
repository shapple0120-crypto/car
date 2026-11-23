"""
FTG + 원호 검사(Arc Check) 반응형 주행기 (v3)

FTG(Follow-the-Gap)만 쓰면 "어디가 뚫렸나"는 알지만 "차가 실제로 그쪽으로 돌아 들어갈 수 있나"는 모른다.
그래서 berlin 헤어핀에서 핸들을 끝까지 꺾어도 회전이 모자라 바깥 벽에 닿았다(harness 장애물 3개: 충돌 0.81회/랩).

이 주행기는 두 단계로 나눈다.
  1) 방향: FTGDriver가 "가장 넓게 뚫린 방향"을 고른다.
  2) 검사: 조향 후보(기본 31개)마다 차가 그릴 원호를 미리 그려서, 벽·장애물에 닿기 전까지 갈 수 있는 거리를 잰다.
     FTG 방향에 가깝고 멀리 갈 수 있는 원호를 고르고, 그 거리 안에서 멈출 수 있는 속도로 달린다.

- 입력은 라이다(scan)뿐이다. x, y, yaw, v, yaw_rate, t는 받기만 하고 쓰지 않는다 (주행 중 GT 미사용).
- 지도, 레이싱라인, 맵 좌표를 쓰지 않는다 (구간별 하드코딩 없음).
- 플래너와 제어기가 따로 있지 않고, 이 클래스 하나가 /drive를 낸다 (MUX 없음).
- 팀 공통 인터페이스: act(scan, x, y, yaw, v, yaw_rate, t) -> (steer, speed), reset()
"""
import numpy as np

from .ftg_driver import FTGDriver  # 같은 drivers 폴더의 FTG 주행기를 '방향 고르기' 부품으로 쓴다


class FTGArcDriver:
    """FTG가 고른 방향 + 원호 검사로 실제로 갈 수 있는 조향과 속도를 정하는 주행기"""

    gt_free = True  # avoid_node.py가 GT 토픽(/car_state/*_GT)을 구독하지 않게 하는 표시

    def __init__(
        self,
        fov_rad: float = 4.7,          # 라이다 시야각 (시뮬레이터: 4.7 rad, 1080빔)
        max_steer: float = 0.4189,     # 최대 조향각 (rad, 약 24도)
        wheelbase: float = 0.33,       # 앞뒤 바퀴 사이 거리 (m). 조향각 -> 회전 곡률 변환에 씀
        lidar_ahead: float = 0.27,     # 라이다가 뒷바퀴 축보다 앞에 달린 거리 (m)
        n_arcs: int = 31,              # 검사할 조향 후보 개수 (왼쪽 끝 ~ 오른쪽 끝을 고르게 나눔)
        arc_len: float = 5.0,          # 원호를 몇 m 앞까지 그려 볼지
        ds: float = 0.1,               # 원호 위 검사 간격 (m)
        front: float = 0.30,           # 검사 기준점: 뒷바퀴 축에서 차 앞부분까지 거리 (m)
        half_width: float = 0.17,      # 차 폭의 절반 (m)
        clear_extra: float = 0.05,     # 차 폭 바깥으로 더 둘 여유 (m)
        free_cap: float = 3.0,         # 갈 수 있는 거리 점수의 상한 (m). 이보다 멀면 똑같이 취급
        w_dir: float = 1.5,            # FTG 방향과 다른 조향에 주는 벌점 무게 (1 rad 차이당 1.5 m 감점)
        w_prev: float = 0.3,           # 직전 조향과 다른 조향에 주는 벌점 무게 (핸들 떨림 방지)
        a_brake: float = 5.0,          # 멈출 때 쓸 감속도 (m/s^2). 속도 상한 계산에 씀
        a_lat: float = 8.0,            # 커브에서 허용할 옆 방향 가속도 (m/s^2)
        d_stop: float = 0.35,          # 막힌 곳 앞 이 거리(m)에서 설 수 있게 속도를 정함
        v_max: float = 7.0,            # 최고 속도 (m/s)
        v_min: float = 0.3,            # 최저 속도 (m/s). 0이면 멈춰서 못 움직이므로 조금 남김
        max_range: float = 10.0,       # 이보다 먼 라이다 값은 잘라서 씀 (m)
    ):
        self.fov = fov_rad
        self.lidar_ahead = lidar_ahead
        self.arc_len = arc_len
        self.front = front
        self.clear2 = (half_width + clear_extra) ** 2   # 필요한 여유 거리의 제곱 (제곱끼리 비교하면 sqrt 계산을 아낌)
        self.free_cap, self.w_dir, self.w_prev = free_cap, w_dir, w_prev
        self.a_brake, self.a_lat, self.d_stop = a_brake, a_lat, d_stop
        self.v_max, self.v_min, self.max_range = v_max, v_min, max_range

        # 방향을 고를 FTG. 원호 검사가 떨림을 잡으므로 FTG 쪽 스무딩은 끈다(steer_smooth=0)
        self.ftg = FTGDriver(fov_rad=fov_rad, max_steer=max_steer, steer_smooth=0.0)

        # 조향 후보와 각 후보의 곡률(1/회전 반지름). 자전거 모델: 곡률 = tan(조향각) / 축간 거리
        self.steers = np.linspace(-max_steer, max_steer, n_arcs)
        self.kappa = np.tan(self.steers) / wheelbase

        # 각 원호 위 검사 지점의 좌표를 미리 계산해 둔다 (뒷바퀴 축 기준, x는 앞, y는 왼쪽)
        self.s = np.arange(ds, arc_len + 1e-9, ds)                     # 원호를 따라간 거리들
        k = self.kappa[:, None]                                         # (후보 수, 1)
        s = self.s[None, :]                                             # (1, 지점 수)
        straight = np.abs(k) < 1e-6                                     # 곡률이 거의 0이면 직선
        k_safe = np.where(straight, 1.0, k)                             # 0으로 나누기 방지용
        x = np.where(straight, s, np.sin(k_safe * s) / k_safe)          # 원 위를 s만큼 갔을 때 x
        y = np.where(straight, 0.0, (1.0 - np.cos(k_safe * s)) / k_safe)  # 원 위를 s만큼 갔을 때 y
        heading = k * s                                                 # 그 지점에서 차가 향하는 각도
        # 차 앞부분(front)이 지나가는 자리 = 뒷바퀴 축 위치 + 진행 방향으로 front만큼
        self.cx = x + front * np.cos(heading)
        self.cy = y + front * np.sin(heading)

        self.prev = 0.0  # 직전 조향각 기억

    def reset(self):
        """새 에피소드 시작이나 차를 다시 놓았을 때 기억을 지운다"""
        self.prev = 0.0
        self.ftg.reset()

    def _scan_points(self, scan) -> tuple:
        """라이다 거리 -> 뒷바퀴 축 기준 (x, y) 점들. 원호 근처에 있는 점만 남긴다"""
        r = np.asarray(scan, dtype=np.float64)
        r = np.clip(np.nan_to_num(r, nan=0.0, posinf=self.max_range), 0.0, self.max_range)
        n = len(r)
        ang = (np.arange(n) - (n - 1) / 2.0) * (self.fov / (n - 1))    # 각 빔의 각도 (정면 0, 왼쪽 +)
        ok = (r > 0.05) & (r < self.max_range - 0.01)                   # 측정 실패(0)와 '아무것도 없음'(최대값) 제외
        px = r[ok] * np.cos(ang[ok]) + self.lidar_ahead                 # 라이다 기준 -> 뒷바퀴 축 기준으로 옮김
        py = r[ok] * np.sin(ang[ok])
        near = (px > -0.5) & (px < self.arc_len + 1.0) & (np.abs(py) < self.arc_len + 1.0)
        return px[near], py[near]

    def _free_lengths(self, px: np.ndarray, py: np.ndarray) -> np.ndarray:
        """조향 후보마다: 원호를 따라 벽·장애물에 닿기 전까지 갈 수 있는 거리 (m)"""
        free = np.full(len(self.steers), self.arc_len)                   # 아무것도 안 걸리면 끝까지
        if len(px) == 0:
            return free
        # 모든 후보의 모든 검사 지점과 모든 라이다 점 사이 거리를 한 번에 계산 (반복문보다 빠름)
        dx = self.cx.reshape(-1, 1) - px[None, :]                        # (후보 수 x 지점 수, 라이다 점 수)
        dy = self.cy.reshape(-1, 1) - py[None, :]
        hit = (np.min(dx * dx + dy * dy, axis=1) < self.clear2).reshape(self.cx.shape)  # 지점마다 여유가 모자라나?
        any_hit = hit.any(axis=1)                                        # 후보마다 한 번이라도 걸리나?
        first = np.argmax(hit, axis=1)                                   # 처음 걸리는 지점 번호
        free[any_hit] = np.maximum(self.s[first[any_hit]] - 0.1, 0.0)    # 처음 걸리는 지점 조금 앞까지
        return free

    def act(self, scan, x, y, yaw, v, yaw_rate, t):
        """매 제어 주기(50 Hz) 호출. 출력: (조향각 rad, 왼쪽 +), (목표 속도 m/s)"""
        # 1. FTG가 원하는 방향 (FTG의 속도는 쓰지 않는다)
        target, _ = self.ftg.act(scan, x, y, yaw, v, yaw_rate, t)

        # 2. 원호 검사: 조향 후보마다 갈 수 있는 거리
        px, py = self._scan_points(scan)
        free = self._free_lengths(px, py)

        # 3. 점수 = 갈 수 있는 거리(상한 free_cap) - FTG 방향과의 차이 벌점 - 직전 조향과의 차이 벌점
        score = (np.minimum(free, self.free_cap)
                 - self.w_dir * np.abs(self.steers - target)
                 - self.w_prev * np.abs(self.steers - self.prev))
        i = int(np.argmax(score))                                        # 점수가 가장 높은 후보
        steer = float(self.steers[i])
        self.prev = steer

        # 4. 속도 = min(그 거리 안에서 멈출 수 있는 속도, 그 곡률에서 미끄러지지 않는 속도)
        v_brake = np.sqrt(2.0 * self.a_brake * max(free[i] - self.d_stop, 0.0))   # v^2 = 2*a*d
        k = abs(self.kappa[i])
        v_turn = np.sqrt(self.a_lat / k) if k > 1e-3 else self.v_max             # v^2 * 곡률 = 옆 가속도
        speed = float(np.clip(min(v_brake, v_turn), self.v_min, self.v_max))
        return steer, speed
