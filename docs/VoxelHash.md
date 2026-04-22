# VoxelHash 수학적 정식화 (`Feature/VoxelHash.cpp`)

## 1. 표기와 데이터 구조

복셀 키를 정수 격자 좌표로 둔다.
\[
\mathbf{k}=(k_x,k_y,k_z)\in\mathbb{Z}^3,\quad 
\mathbf{k}=\left\lfloor \frac{\mathbf{p}}{s}\right\rfloor
\]
- \(\mathbf{p}\in\mathbb{R}^3\): 샘플 포인트
- \(s>0\): voxel size (`voxelSize`)

해시 테이블은 버킷 기반:
- 버킷 수 \(B\) (`VH_NUM_BUCKETS`)
- 버킷당 슬롯 수 \(S=4\) (`BUCKET_SIZE`)
- 전체 엔트리 수 \(N=B\cdot S\)

각 엔트리(슬롯)는 다음 상태를 가진다:
\[
(k_x,k_y,k_z,\; d,\; w,\; c,\; n,\; a_d,\; a_w,\; a_c,\; a_n,\; f)
\]
- \(d\): 확정 TSDF
- \(w\): 확정 weight
- \(c\): color
- \(n\): normal(Oct encoding)
- \(a_d, a_w, a_c, a_n\): 이번 배치 누산(accumulator)
- \(f\): 마지막 업데이트 프레임 tag

---

## 2. 공간 해시와 버킷 매핑

코드의 해시:
\[
h(\mathbf{k})=
\Big((k_x\cdot 73856093)\oplus(k_y\cdot 19349663)\oplus(k_z\cdot 83492791)\Big)\bmod B
\]
(\(\oplus\): bitwise XOR)

CPU에서 포인트를 \(h(\mathbf{k})\) 기준 정렬한 뒤, 각 버킷의 구간
\[
[\texttt{cellStart}[b],\ \texttt{cellEnd}[b])
\]
을 만든다.  
즉 버킷 \(b\)를 담당하는 workgroup은 이 구간의 포인트만 순회한다.

---

## 3. Gather P2G 커널의 수학적 의미

## 3.1 슬롯 할당 (버킷 내부 키 집합 형성)
버킷 \(b\)의 샘플 집합:
\[
\mathcal{P}_b=\{i\mid i\in[\texttt{cellStart}[b],\texttt{cellEnd}[b))\}
\]
각 샘플의 키 \(\mathbf{k}_i\)를 버킷 내 최대 4개 슬롯에 배정한다.
- 이미 같은 키 슬롯이 있으면 reuse
- 없고 빈 슬롯이 있으면 새 키 삽입
- 슬롯이 가득 차면 해당 키는 이 배치에서 반영되지 않음(암묵적 drop)

---

## 3.2 샘플별 TSDF/weight 계산

센서 위치 \(\mathbf{s}\), 포인트 \(\mathbf{p}_i\), 키 \(\mathbf{k}_i\), truncation \(\tau\).

1) 시선 기반 법선 근사:
\[
\mathbf{n}_i=\frac{\mathbf{s}-\mathbf{p}_i}{\|\mathbf{s}-\mathbf{p}_i\|}
\]

2) 해당 복셀 중심:
\[
\mathbf{v}_i=(\mathbf{k}_i+0.5)\,s
\]

3) TSDF 샘플:
\[
\tilde d_i=\frac{(\mathbf{v}_i-\mathbf{p}_i)\cdot \mathbf{n}_i}{\tau},\quad
d_i=\mathrm{clamp}(\tilde d_i,-1,1)
\]

4) 가중치:
\[
w_i=\max\!\left(\mathbf{n}_i\cdot\frac{\mathbf{s}-\mathbf{v}_i}{\|\mathbf{s}-\mathbf{v}_i\|},\ 0.05\right)
\]

---

## 3.3 슬롯별 누산 (원자 연산 없는 gather-reduce)

슬롯 \(s\)에 매칭된 샘플 집합을 \(\mathcal{I}_s\)라 하면, 목표 누산량:
\[
A_d^{(s)}=\sum_{i\in\mathcal{I}_s} d_i\,w_i,\qquad
A_w^{(s)}=\sum_{i\in\mathcal{I}_s} w_i
\]
코드는 thread-local 누산 후 shared memory tree reduction으로 위 합을 계산한다.

색/법선은 합산이 아니라 최신값 선택(last-write-like):
\[
A_c^{(s)}\leftarrow c_{i^\*},\quad A_n^{(s)}\leftarrow n_{i^\*}
\]
(\(i^\*\): 구현상 선택된 마지막 관측 샘플)

---

## 4. Finalize 커널: running weighted average

점유 슬롯(키가 EMPTY 아님)이고 \(A_w>0\)인 경우:
- 이전 상태 \((d_{\text{old}},w_{\text{old}})\)
- 이번 배치 누산 \((A_d,A_w)\)

갱신식:
\[
w_{\text{new}}=\min(w_{\text{old}}+A_w,\ 50.0)
\]
\[
d_{\text{new}}=\mathrm{clamp}\!\left(
\frac{d_{\text{old}}\,w_{\text{old}}+A_d}{w_{\text{old}}+A_w},\ -1,\ 1
\right)
\]

그리고:
\[
c\leftarrow A_c,\quad n\leftarrow A_n,\quad f\leftarrow \texttt{currentFrame}
\]
\[
A_d,A_w,A_c,A_n \leftarrow 0
\]

> 주의: C++의 `maxWeight_` UI 파라미터가 있어도, 현재 finalize GLSL은 상수 `50.0`으로 clamp한다.

---

## 5. Count 커널

점유 슬롯 수:
\[
\text{occupancy}=\sum_{j=0}^{N-1}\mathbf{1}[k_x^{(j)}\neq \text{EMPTY}]
\]
를 원자 증가로 집계해 통계를 만든다.

---

## 6. 복잡도/병목 관점 요약

- CPU 정렬 + 인덱스 구축:
  - 정렬 \(O(P\log P)\), 경계 생성 \(O(P)\), \(P=\) 포인트 수
- Gather P2G:
  - 버킷별 병렬 처리, 버킷 내부는 최대 4개 슬롯에 대한 reduction
- 설계 핵심:
  - **scatter + global atomic** 대신
  - **bucket-owned workgroup + local reduction + atomic-free write-back**
- 트레이드오프:
  - 버킷 슬롯 수 4의 한계로 고밀도 충돌 시 drop 가능
  - 코드에서 `maxPtsPerBucket` 제한으로 과밀 버킷을 추가적으로 절단 가능

---

## 7. 코드-수식 대응 포인트(빠른 참조)

- 해시 함수: `hashBucket(...)`, GLSL `hash3`
- TSDF/weight 식: gather shader `tsdf`, `w` 계산
- reduce: `REDUCE_SLOT` 매크로
- finalize 평균식: `kVH_FinalizeComp`의 `newT`, `newW`
- occupancy: `kVH_CountComp`
- 시각화 recency: `frame_tag` 기반 highlight