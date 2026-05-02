import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ================== 한글 폰트 설정 ==================
plt.rcParams['font.family'] = 'Malgun Gothic'      # Windows 사용자
# plt.rcParams['font.family'] = 'NanumGothic'      # Mac이나 Linux에서 NanumGothic이 설치된 경우
plt.rcParams['axes.unicode_minus'] = False         # 마이너스 기호 깨짐 방지
plt.rcParams['font.family'] = 'AppleGothic'

# ================== 각속도 애니메이션 ==================
omega = 2.0          # 각속도 (rad/s) ← 여기 숫자 바꿔가며 테스트해보세요!
duration = 10        # 애니메이션 길이 (초)
fps = 30

fig, ax = plt.subplots(figsize=(9, 9))
ax.set_xlim(-1.3, 1.3)
ax.set_ylim(-1.3, 1.3)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

ax.plot(0, 0, 'ko', markersize=10, label='회전 중심')

line, = ax.plot([], [], 'b-', linewidth=7, label='위치 벡터')
arrow = ax.arrow(0, 0, 0, 0, head_width=0.08, head_length=0.12, 
                 fc='red', ec='red', lw=2.5)

time_text = ax.text(-1.25, 1.18, '', fontsize=13)
angle_text = ax.text(-1.25, 1.02, '', fontsize=13)
omega_text = ax.text(0.4, 1.18, f'각속도 ω = {omega:.1f} rad/s', 
                     fontsize=14, color='darkblue', fontweight='bold')

ax.set_title('각속도(ω) 직관적 이해\nθ = ω × t', fontsize=18, pad=25)
ax.legend(loc='upper right')

def init():
    line.set_data([], [])
    return line, arrow, time_text, angle_text

def animate(frame):
    t = frame / fps
    theta = omega * t
    
    x = np.cos(theta)
    y = np.sin(theta)
    
    line.set_data([0, x], [0, y])
    arrow.set_data(x=0, y=0, dx=0.92*x, dy=0.92*y)
    
    angle_deg = np.degrees(theta) % 360
    time_text.set_text(f'시간  t  = {t:.2f} 초')
    angle_text.set_text(f'각도 θ = {angle_deg:.1f}°   ({theta:.2f} rad)')
    
    return line, arrow, time_text, angle_text

ani = FuncAnimation(fig, animate, frames=int(duration * fps),
                    init_func=init, interval=1000/fps, blit=False, repeat=True)

plt.figtext(0.5, 0.03, 
            '파란 선 = 물체의 현재 위치    |    빨간 화살표 = 회전 방향\n'
            'ω 값을 0.5, 1.0, 3.0, 6.28 등으로 바꿔보세요!', 
            ha='center', fontsize=12, bbox=dict(boxstyle="round", facecolor="#f0f0f0"))

plt.tight_layout()
plt.show()