"""
FTG + 원호 검사(Arc Check) 반응형 주행기 (v3.1)

FTG(Follow-the-Gap)만 쓰면 "어디가 뚫렸나"는 알지만 "차가 실제로 그쪽으로 돌아 들어갈 수 있나"는 모른다.
그래서 berlin 헤어핀에서 핸들을 끝까지 꺾어도 회전이 모자라 바깥 벽에 닿았다(harness 장애물 3개: 충돌 0.81회/랩).

이 주행기는 두 단계로 나눈다.
  1) 방향: FTGDriver가 가려는 방향을 고른다. v3.1부터는 "가장 넓은" 빈틈이 아니라 "가장 깊은 점을 품은" 빈틈을 고른다.
     (berlin s ≈ 24 m: 왼쪽으로 꺾이는 좁고 깊은 길 대신, 정면의 넓고 얕은 막다른 곳으로 들어가 U턴·역주행하던 문제 해결)
  2) 검사: 조향 후보(기본 31개)마다 차가 그릴 길을 미리 그려서, 벽·장애물에 닿기 전까지 갈 수 있는 거리를 잰다.
     v3.1부터는 후보 원호 앞에 '전환 구간'을 붙인다: 명령 지연(30 ms) 동안 지금 조향으로, 서보가 새 조향까지
     도는 동안(3.2 rad/s) 중간 조향으로 달리는 길. 빠를수록·크게 꺾을수록 전환 구간이 길어져, 늦은 급회피가 불리해진다.
     FTG 방향에 가깝고 멀리 갈 수 있는 길을 고르고, 그 거리 안에서 멈출 수 있는 속도로 달린다.

- 입력은 라이다(scan)뿐이다. x, y, yaw, v, yaw_rate는 쓰지 않는다 (GT-free 모드의 ROS 노드는 NaN을 넘긴다).
  지금 속도와 조향각은 '내가 보낸 명령'과 시뮬레이터 한계(서보 속도, 가속 한계)로 추정한다.
- 지도, 레이싱라인, 맵 좌표를 쓰지 않는다 (구간별 하드코딩 없음).
- 플래너와 제어기가 따로 있지 않고, 이 클래스 하나가 /drive를 낸다 (MUX 없음).
- 팀 공통 인터페이스: act(scan, x, y, yaw, v, yaw_rate, t) -> (steer, speed), reset()
"""
import numpy as np
from scipy.ndimage import distance_transform_edt  # 격자에서 '가장 가까운 장애물 칸까지 거리'를 한 번에 구하는 함수

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
        grid_res: float = 0.04,        # 원호 검사용 지역 격자 한 칸 크기 (m)
        gap_mode: str = "deepest",     # FTG 빈틈 고르는 기준: 가장 깊은 점을 품은 빈틈 (막다른 곳 회피, ftg_driver.py 참고)
        gap_keep: float = 0.0,         # FTG가 직전 빈틈을 유지하도록 주는 가산점 비율
        transition: bool = True,       # True: 조향이 바뀌는 동안(지연 + 서보 회전) 차가 실제로 가는 길까지 검사
        latency: float = 0.03,         # 명령이 차에 닿기까지 걸리는 시간 (s)
        steer_rate: float = 3.2,       # 조향 서보가 돌 수 있는 최대 속도 (rad/s, 시뮬레이터 값)
        accel_max: float = 9.5,        # 속도가 바뀔 수 있는 최대 가속도 (m/s^2), 내 속도 추정에 씀
        n_pre: int = 6,                # 전환 구간 검사 지점 수
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
        self.ftg = FTGDriver(fov_rad=fov_rad, max_steer=max_steer, steer_smooth=0.0, gap_mode=gap_mode, gap_keep=gap_keep)

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

        # 원호 검사용 지역 격자: 뒷바퀴 축 기준 x는 -0.5 ~ arc_len+0.5 m, y는 ±(arc_len+0.5) m
        self.res = grid_res
        self.gx0, self.gy0 = -0.5, -(arc_len + 0.5)                     # 격자 왼쪽 아래 모서리 좌표
        self.nx = int(np.ceil((arc_len + 1.0) / grid_res))              # x 방향 칸 수
        self.ny = int(np.ceil((2 * arc_len + 1.0) / grid_res))          # y 방향 칸 수
        # 원호 위 검사 지점들이 격자의 몇 번째 칸인지 미리 계산 (격자 밖은 -1)
        ix = np.floor((self.cx - self.gx0) / grid_res).astype(int)
        iy = np.floor((self.cy - self.gy0) / grid_res).astype(int)
        inside = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        self.ix, self.iy, self.inside = np.where(inside, ix, 0), np.where(inside, iy, 0), inside

        self.transition, self.latency, self.steer_rate = transition, latency, steer_rate
        self.accel_max, self.n_pre, self.wheelbase = accel_max, n_pre, wheelbase
        # 내가 보낸 명령으로 추정하는 지금의 실제 조향각·속도 (GT-free 모드에서는 v가 NaN으로 들어오므로)
        self.est_delta, self.est_v, self.cmd_speed, self.last_t = 0.0, 0.0, 0.0, None

        self.prev = 0.0  # 직전 조향각 기억

    def reset(self):
        """새 에피소드 시작이나 차를 다시 놓았을 때 기억을 지운다"""
        self.prev = 0.0
        self.ftg.reset()
        self.est_delta, self.est_v, self.cmd_speed, self.last_t = 0.0, 0.0, 0.0, None

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
        """조향 후보마다: 원호를 따라 벽·장애물에 닿기 전까지 갈 수 있는 거리 (m)

        방법: 라이다 점을 작은 격자(4 cm 칸)에 찍고, 거리 변환으로 '각 칸에서 가장 가까운 점까지 거리' 지도를 만든다.
        그다음 원호 위 검사 지점의 칸에서 그 값을 읽기만 하면 된다 (점마다 거리를 일일이 재는 것보다 훨씬 빠름).
        """
        free = np.full(len(self.steers), self.arc_len)                   # 아무것도 안 걸리면 끝까지
        if len(px) == 0:
            return free
        occ = np.zeros((self.nx, self.ny), dtype=bool)                   # 격자: True = 라이다 점이 있는 칸
        ix = np.floor((px - self.gx0) / self.res).astype(int)
        iy = np.floor((py - self.gy0) / self.res).astype(int)
        ok = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        occ[ix[ok], iy[ok]] = True
        if not occ.any():
            return free
        dist = distance_transform_edt(~occ) * self.res                   # 각 칸에서 가장 가까운 점 칸까지 거리 (m)
        d = np.where(self.inside, dist[self.ix, self.iy], np.inf)        # 원호 위 검사 지점마다 여유 거리
        hit = d < np.sqrt(self.clear2)                                   # 여유가 모자란 지점
        any_hit = hit.any(axis=1)                                        # 후보마다 한 번이라도 걸리나?
        first = np.argmax(hit, axis=1)                                   # 처음 걸리는 지점 번호
        free[any_hit] = np.maximum(self.s[first[any_hit]] - 0.1, 0.0)    # 처음 걸리는 지점 조금 앞까지
        return free

    @staticmethod
    def _arc_pose(x0, y0, h0, k, s):
        """(x0, y0, 방향 h0)에서 곡률 k로 거리 s만큼 갔을 때의 위치와 방향 (배열 연산 가능)"""
        small = np.abs(k) < 1e-6
        ks = np.where(small, 1.0, k)
        x = np.where(small, x0 + s * np.cos(h0), x0 + (np.sin(h0 + ks * s) - np.sin(h0)) / ks)
        y = np.where(small, y0 + s * np.sin(h0), y0 + (np.cos(h0) - np.cos(h0 + ks * s)) / ks)
        return x, y, h0 + np.where(small, 0.0, k * s)

    def _free_lengths_transition(self, px, py):
        """조향 후보마다 갈 수 있는 거리. 단, 후보 조향에 닿기 전 '전환 구간'을 먼저 지난다고 보고 검사한다.

        전환 구간 = 명령 지연(latency) 동안 지금 조향으로 + 서보가 새 조향까지 도는 동안 중간 조향으로 달리는 길.
        속도가 빠르고 조향 차이가 클수록 전환 구간이 길어진다 → 늦게 꺾는 선택은 자연스럽게 불리해진다.
        """
        n = len(self.steers)
        free = np.full(n, self.arc_len)
        occ = np.zeros((self.nx, self.ny), dtype=bool)
        ix = np.floor((px - self.gx0) / self.res).astype(int)
        iy = np.floor((py - self.gy0) / self.res).astype(int)
        ok = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        occ[ix[ok], iy[ok]] = True
        if not occ.any():
            return free
        dist = distance_transform_edt(~occ) * self.res
        v = max(self.est_v, 0.0)
        # 1) 지연 구간: 지금 조향(est_delta)으로 v * latency 만큼
        k_now = np.tan(self.est_delta) / self.wheelbase
        l_lat = v * self.latency
        # 2) 서보 구간: 중간 조향으로 v * |차이| / steer_rate 만큼
        d_delta = np.abs(self.steers - self.est_delta)
        l_ramp = v * d_delta / self.steer_rate
        k_mid = np.tan(0.5 * (self.steers + self.est_delta)) / self.wheelbase
        x1, y1, h1 = self._arc_pose(0.0, 0.0, 0.0, k_now, l_lat)                  # 지연 구간 끝 (후보 공통)
        x2, y2, h2 = self._arc_pose(x1, y1, h1, k_mid, l_ramp)                    # 서보 구간 끝 (후보마다)
        # 전환 구간 검사 지점들 (차 앞부분 위치)
        f = np.linspace(0.0, 1.0, self.n_pre)[None, :]                            # (1, n_pre)
        L_pre = l_lat + l_ramp                                                    # (n,)
        s_pre = f * L_pre[:, None]                                                # (n, n_pre)
        in_lat = s_pre <= l_lat
        xa, ya, ha = self._arc_pose(0.0, 0.0, 0.0, k_now, np.minimum(s_pre, l_lat))
        xb, yb, hb = self._arc_pose(x1, y1, h1, k_mid[:, None], np.maximum(s_pre - l_lat, 0.0))
        xp = np.where(in_lat, xa, xb); yp = np.where(in_lat, ya, yb); hp = np.where(in_lat, ha, hb)
        xp = xp + self.front * np.cos(hp); yp = yp + self.front * np.sin(hp)
        # 후보 원호(앞부분 위치, 미리 계산한 cx, cy)를 서보 구간 끝 자세로 옮긴다
        c, sn = np.cos(h2)[:, None], np.sin(h2)[:, None]
        xc = x2[:, None] + c * self.cx - sn * self.cy
        yc = y2[:, None] + sn * self.cx + c * self.cy
        def clearance(xq, yq):
            gx = np.floor((xq - self.gx0) / self.res).astype(int)
            gy = np.floor((yq - self.gy0) / self.res).astype(int)
            inside = (gx >= 0) & (gx < self.nx) & (gy >= 0) & (gy < self.ny)
            return np.where(inside, dist[np.clip(gx, 0, self.nx - 1), np.clip(gy, 0, self.ny - 1)], np.inf)
        r_need = np.sqrt(self.clear2)
        hit_pre = clearance(xp, yp) < r_need                                      # (n, n_pre)
        hit_arc = clearance(xc, yc) < r_need                                      # (n, 지점 수)
        for i in range(n):
            hp_i = np.flatnonzero(hit_pre[i])
            if len(hp_i):
                free[i] = max(s_pre[i, hp_i[0]] - 0.1, 0.0)                      # 전환 중에 이미 부딪힘
                continue
            ha_i = np.flatnonzero(hit_arc[i])
            free[i] = L_pre[i] + (self.s[ha_i[0]] - 0.1 if len(ha_i) else self.arc_len)
        return free

    def _update_estimate(self, t):
        """직전 명령으로 지금의 실제 조향각과 속도를 추정한다 (서보 속도·가속 한계를 따름)"""
        dt = 0.02 if (self.last_t is None or not np.isfinite(t)) else float(np.clip(t - self.last_t, 0.005, 0.1))
        self.last_t = t if np.isfinite(t) else None
        step = self.steer_rate * dt
        self.est_delta += float(np.clip(self.prev - self.est_delta, -step, step))
        dv = self.accel_max * dt
        self.est_v += float(np.clip(self.cmd_speed - self.est_v, -dv, dv))

    def act(self, scan, x, y, yaw, v, yaw_rate, t):
        """매 제어 주기(50 Hz) 호출. 출력: (조향각 rad, 왼쪽 +), (목표 속도 m/s)"""
        # 1. FTG가 원하는 방향 (FTG의 속도는 쓰지 않는다)
        target, _ = self.ftg.act(scan, x, y, yaw, v, yaw_rate, t)

        # 2. 원호 검사: 조향 후보마다 갈 수 있는 거리
        px, py = self._scan_points(scan)
        if self.transition:
            self._update_estimate(t)
            free = self._free_lengths_transition(px, py)
        else:
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
        self.cmd_speed = speed
        return steer, speed
