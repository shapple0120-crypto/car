"""간이 시뮬레이터 (동작 확인용): 공개 f1tenth_gym 물리 + berlin 맵, ROS·race_manager 불필요.

장애물(0.3 m 정사각형)을 레이싱라인 근처에 무작위로 지도 이미지에 그려 넣고, 충돌하면 그 에피소드를 끝낸다.
명령 지연 30 ms, 라이다 0.27 m 앞, 라이다 중심 0.33 x 0.20 m 충돌 박스로 harness 조건을 흉내 낸다.
race_manager 배치 규칙과 다르므로 harness·ROS 결과와 숫자를 섞어 비교하면 안 된다.

준비:
  git clone https://github.com/f1tenth/f1tenth_gym.git   (gym 패키지는 설치하지 않아도 됨)
  pip install numpy scipy numba pillow pyyaml
사용:
  F110_GYM=~/f1tenth_gym MAP_DIR=~/forza_ws/race_stack/stack_master/maps/berlin \
  python3 tools/mini_sim.py . drivers.ftg_driver:FTGDriver 1,3,5 8 '{}'
  인자: <주행기 폴더> <module:Class> <장애물 수 목록> <seed 개수> ['<json kwargs>']
출력: 에피소드마다 json 한 줄 (laps=연속 무충돌 랩, crash, kind=wall/obstacle, best, times)
"""
import sys, types, json, os, math, importlib
import numpy as np
from PIL import Image
import yaml
# f110_gym/__init__가 gym을 import하므로 빈 껍데기 모듈로 대신한다
g = types.ModuleType('gym'); g.envs = types.ModuleType('gym.envs'); g.envs.registration = types.ModuleType('reg')
g.envs.registration.register = lambda *a, **k: None
sys.modules.update({'gym': g, 'gym.envs': g.envs, 'gym.envs.registration': g.envs.registration})
GYM = os.path.expanduser(os.environ.get('F110_GYM', '~/f1tenth_gym')) + '/gym'
sys.path.insert(0, GYM)
import importlib.util
pkg = types.ModuleType('f110_gym'); pkg.__path__ = [GYM + '/f110_gym']; sys.modules['f110_gym'] = pkg
envs = types.ModuleType('f110_gym.envs'); envs.__path__ = [pkg.__path__[0] + '/envs']; sys.modules['f110_gym.envs'] = envs
from f110_gym.envs.base_classes import RaceCar

MAPDIR = os.path.expanduser(os.environ.get('MAP_DIR', '~/forza_ws/race_stack/stack_master/maps/berlin'))
P = dict(mu=1.0489, C_Sf=4.718, C_Sr=5.4562, lf=0.15875, lr=0.17145, h=0.074, m=3.74, I=0.04712, s_min=-0.4189,
         s_max=0.4189, sv_min=-3.2, sv_max=3.2, v_switch=7.319, a_max=9.51, v_min=-5.0, v_max=20.0, width=0.31, length=0.58)

def run(driver_spec, seed, n_obs, laps=10, kw=None, max_t=180.0, outdir=None, latency=0.03, lidar_dist=0.27):
    outdir = outdir or f'/tmp/simmaps_{os.getpid()}'; os.makedirs(outdir, exist_ok=True)
    meta = yaml.safe_load(open(f'{MAPDIR}/berlin.yaml'))
    res, ox, oy = meta['resolution'], meta['origin'][0], meta['origin'][1]
    img = np.array(Image.open(f'{MAPDIR}/berlin.png').convert('L'))
    H = img.shape[0]
    wp = json.load(open(f'{MAPDIR}/global_waypoints.json'))['global_traj_wpnts_sp']['wpnts']
    xs = np.array([w['x_m'] for w in wp]); ys = np.array([w['y_m'] for w in wp]); ss = np.array([w['s_m'] for w in wp])
    L = ss[-1] + math.hypot(xs[0]-xs[-1], ys[0]-ys[-1])
    rng = np.random.default_rng(seed)
    # 장애물: 출발선 5 m 이후 레이싱라인 근처(좌우 ±0.6 m)에 0.3 m 정사각형
    cand = np.nonzero((ss > 5) & (ss < L - 3))[0]
    chosen = []
    while len(chosen) < n_obs:
        i = int(rng.choice(cand))
        if all(abs(ss[i]-ss[j]) > 3.0 for j in chosen): chosen.append(i)
    m = img.copy()
    for i in chosen:
        hd = math.atan2(ys[(i+1) % len(ys)]-ys[i], xs[(i+1) % len(xs)]-xs[i])
        d = rng.uniform(-0.6, 0.6)
        cx, cy = xs[i] - d*math.sin(hd), ys[i] + d*math.cos(hd)
        for a in np.linspace(-0.15, 0.15, 12):
            for b in np.linspace(-0.15, 0.15, 12):
                px = int((cx + a - ox)/res); py = H - 1 - int((cy + b - oy)/res)
                m[py, px] = 0
    name = f'{outdir}/b_{seed}_{n_obs}'
    Image.fromarray(m).save(name + '.png')
    meta2 = dict(meta); meta2['image'] = os.path.basename(name) + '.png'
    yaml.safe_dump(meta2, open(name + '.yaml', 'w'))
    from scipy.ndimage import distance_transform_edt as edt
    dist = edt(m > 128) * res   # 각 픽셀에서 가장 가까운 벽/장애물까지 거리
    car = RaceCar(P, seed, time_step=0.01, num_beams=1080, fov=4.7, lidar_dist=lidar_dist)
    car.set_map(name + '.yaml', '.png')
    yaw0 = math.atan2(ys[1]-ys[0], xs[1]-xs[0])
    car.reset(np.array([xs[0], ys[0], yaw0]))
    mod, cls = driver_spec.split(':'); drv = getattr(importlib.import_module(mod), cls)(**(kw or {})); drv.reset()
    scan = car.update_pose(0.0, 0.0)
    prev_i, lap_start, lap_times, t, steer, speed = 0, 0.0, [], 0.0, 0.0, 0.0
    prog = 0.0
    queue = []
    while t < max_t and len(lap_times) < laps:
        if round(t / 0.01) % 2 == 0:
            st = car.state
            queue.append((t + latency, drv.act(scan, st[0], st[1], st[4], st[3], st[5], t)))   # 명령은 latency 뒤에 적용
        while queue and queue[0][0] <= t + 1e-9:
            steer, speed = queue.pop(0)[1]
        scan = car.update_pose(steer, speed); t += 0.01
        hd = car.state[4]
        cx_ = car.state[0] + lidar_dist*math.cos(hd); cy_ = car.state[1] + lidar_dist*math.sin(hd)   # 라이다 위치
        geo_hit = False   # 라이다 중심 0.33 x 0.20 m 박스의 9개 점 중 하나라도 벽·장애물 안이면 충돌
        for a_ in (-0.165, 0.0, 0.165):
            for b_ in (-0.10, 0.0, 0.10):
                qx = cx_ + a_*math.cos(hd) - b_*math.sin(hd); qy = cy_ + a_*math.sin(hd) + b_*math.cos(hd)
                if dist[H-1-int((qy-oy)/res), int((qx-ox)/res)] < 0.02: geo_hit = True
        if car.in_collision or geo_hit:
            d0 = edt(img > 128)*res
            kind = 'wall' if d0[H-1-int((cy_-oy)/res), int((cx_-ox)/res)] < 0.3 else 'obstacle'
            return dict(kind=kind, v=round(float(car.state[3]),2), s=round(float(ss[int(np.argmin((xs-car.state[0])**2+(ys-car.state[1])**2))]),1), seed=seed, obs=n_obs, laps=len(lap_times), crash=True, t=round(t,2), best=min(lap_times) if lap_times else None, times=lap_times)
        i = int(np.argmin((xs-car.state[0])**2 + (ys-car.state[1])**2))
        ds = ss[i] - ss[prev_i]
        if ds < -L/2: ds += L
        if ds > L/2: ds -= L
        prog += ds; prev_i = i
        if prog >= L * (len(lap_times) + 1):
            lap_times.append(round(t - lap_start, 2)); lap_start = t
    return dict(seed=seed, obs=n_obs, laps=len(lap_times), crash=False, t=round(t,2), best=min(lap_times) if lap_times else None, times=lap_times)

if __name__ == '__main__':
    sys.path.insert(0, sys.argv[1])
    spec_ = sys.argv[2]; kw = json.loads(sys.argv[5]) if len(sys.argv) > 5 else {}
    for n in [int(v) for v in sys.argv[3].split(',')]:
        for s in range(int(sys.argv[4])):
            print(json.dumps(run(spec_, s, n, kw=kw)), flush=True)
