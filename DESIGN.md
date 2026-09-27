# T-001 — 用 m-mode 观测模型评价 global 21 cm horn beam

状态：已实现可安装 package 与合成示例；真实 2025 JBO 协议输入尚未冻结，未计算科学候选 beam 排名。

## 当前实现（2026-09-27）

用户确认可以让输入端直接提交 beam 球谐。主接口因此改为 `BeamModes`：
healpy packed `full_alm[frequency, alm]`，代表 beam-local 全天 Stokes-I **功率增益**。
`BeamModes.prepare(latitude_deg)` 一次性计算地平线加权后的本地 `sky_alm`、
基准 LST 赤道坐标 `sky_ref_alm`、全天辐射效率 `eta_rad` 和物理地面分割
`h_partition`。这些量可直接随候选输入，供生成式搜索跳过逐候选的
HEALPix 变换和 Wigner 旋转。`BeamPattern` 的 theta/phi dBi 网格只作为可选转换器。

`HarmonicScorer` 为给定的天空和 band-limit 预计算 **quadrature sky alms**；
每候选按 limTOD `mmode-solver` 的 packed-alm 公式计算所有 `m`，合成 360 个
LST bin 的 TOD，再计算逐 bin 热噪声与 `d₀`。此环境的 JAX 和 healpy OpenMP
运行库在同一进程冲突，因此热路径使用 NumPy/healpy 实现同一代数。独立慢路径
`reference_modes_forward` 调用 limTOD 的逐 LST beam 旋转和 `mmode.solve`；
等温及非均匀访问、各向异性天空的测试均通过。参见 [README.md](README.md)。
地平线正中的 HEALPix 像素按半天空、半地面计数；这不同于 limTOD 可选的
严格零权 horizon mask，却避免有限 nside 下两个温度项不守恒。预处理后的
`sky_alm` 直接交给相同的 m-mode 代数，不再次施加上游可选遮罩。

下文的实验协议和 FOM 数学定义仍然适用。原先的 zonal grid 快路径现为诊断选项，
其 LST 平均 `Tsys` 噪声近似不用于主排名。真实天空 cube、平滑 21 cm 模板及
2025 访问数未打包；`--require-reference-band` 只验证频率网格，不能认证这些输入。

## 目标与上游边界

给定一个接收端口在各频率、全天方向的 Stokes-I 功率增益球谐系数，返回一个可用于生成式设计搜索的单值分数。θ/φ dBi 网格可在输入端自行转换。分数衡量：在固定站点、观测时长、天空、前景模型和 21 cm 模板下，`d₀(ν)` 中可保留多少信号，同时要付出多少热噪声与前景残差代价。它不是对真实实验检出的保证。

依赖 limTOD 的 `mmode-solver` 分支。上游 T-006 是 RHINO `d₀` 研究的本地网站与证据索引；数值定义主要来自同一分支的 T-002 及全年夜间 LST 后续研究。T-002 说明 8 h 连续弧的未建模高阶模会严重污染 `d₀`；全年夜间堆叠覆盖所有 LST 后，闭合扫描的 `d₀` 就是 LST 平均。不能把前者的 `d₀` 估计误差当成可通过优化 beam 普遍消除的误差。

## 固定基准实验（v1）

| 量 | 固定值与理由 |
| --- | --- |
| 地点/指向 | Jodrell Bank，纬度 53.23625°、经度 −2.30744°、天顶漂移扫描；直接承接上游配置。 |
| 时间 | 2025 年每 240 s、太阳高度 < 0° 的夜间采样，累计到 360 个 1° LST bin；上游记录各 bin 均有覆盖。候选之间使用同一组 bin 和访问数。 |
| 频段 | 55–120 MHz，1 MHz 通道。上游宽频试验说明 55–85 MHz 对前景与 fiducial 信号的区分很弱；55–120 MHz 的真实色散 beam 尚需新验证。要求输入覆盖全部通道，不外推。 |
| 天空 | 冻结两个独立的赤道坐标前景 cube：CNN-PL + GLEAM 和 GSM2008 + GLEAM；记录原始版本/哈希、像素化及频率插值。对平滑插值的人为谱结构做专项检查。 |
| 信号 | 一条冻结、**先按 1 MHz 通道响应平均**的平滑 fiducial `T₂₁(ν)`；可从上游 21cmVAE fiducial 构造，但先移除其节点线性插值造成的伪高频折点，并缓存结果。模板振幅 `A=1`。 |
| 接收模型 | 单端口、未偏振的 Stokes I 天空；固定 `T_rx=100 K`，地面/欧姆损耗物理温度 300 K；独立热噪声，`Δν=1 MHz`。暂不计 ionosphere、RFI、日间标定漂移、天空偏振泄漏。 |
| 前景拟合 | 预先冻结一种平滑前景族，例如正温度的 `exp[Σ_{k=0}^5 a_k ln(ν/70 MHz)^k]`；所有候选与天空场景共用阶数、频段和约束。先用参考 horn/无色散基线验证阶数与残差，不以候选结果反向调阶。 |

这些数值是**评分协议**，不是自然常数；改动时提升协议版本，旧分数不可直接比较。另提供诊断配置用于 55–85 MHz、8 h 弧、其他站点和观测策略，但不混入 v1 主排行榜。

## 输入与前向模型

`BeamPattern` 包含频率、`theta`/`phi` 网格、`gain_theta_dbi`、`gain_phi_dbi`、方向坐标约定、方位零点、增益归一化类型，以及可追踪的来源。约定这两列是**同一接收端口**对正交入射极化的功率增益分量，故未偏振天空的总功率响应为 `G = 10^(Gθ/10) + 10^(Gφ/10)`；不可把 dBi 数值直接相加。若数据其实是双端口、相对幅度、场强 dB、directivity 或 realized gain，必须显式声明并转换；不能默默当作 accepted-power gain。仅有两列功率不能确定复 Jones 矩阵，也不能预测偏振前景泄漏。

v1 的绝对温度/噪声模型采用 *accepted-power gain* 和固定完美匹配；全天积分给出 `η_rad(ν)=∫G dΩ/(4π)`。不允许 `η_rad` 显著超出 `[0,1]`；数值积分误差容差另定。高于地平线的响应看天空，低于地平线的响应看 300 K 地面，未辐射部分以 300 K 损耗表示。通道的天线温度为

```text
T_ant(t,ν) = (1/4π) ∫above G(ν,Ω) T_sky(R_t Ω,ν) dΩ
           + (1/4π) ∫below G(ν,Ω) T_ground dΩ
           + [1 − η_rad(ν)] T_loss
           + h(ν) A T₂₁(ν),
h(ν) = (1/4π) ∫above G(ν,Ω) dΩ.
```

需在实现中核对 limTOD 的 beam 归一化、坐标、地平线处理和 dBi 定义；上游 T-002 用 `normalize_beam=True` 且不遮挡地平线，本协议的物理地面模型与其**不同**，因此不能直接复用上游数值结论。若输入只给 normalized pattern，则缺少绝对效率：只能在标明 `shape_only`、固定 `η_rad=1` 的对照模式评分，不得混排。

先按固定夜间采样求每个 LST bin 的平均温度，再得到 `d₀(ν)=360⁻¹Σ_b T_b(ν)`。对应噪声为 `σ²_d₀(ν)=360⁻²Σ_b T_sys,b(ν)²/[Δν × 240 s × n_visits,b]`，其中 `T_sys,b=T_ant,fg,b+T_rx`。对非均匀 bin 覆盖，保留每个 bin 的独立误差；如果未来采用部分 LST 覆盖，必须使用 limTOD 实际估计器和其 covariance，并另外计算高阶模泄漏与条件数。

## 单值 Figure of Merit

将信号关闭，求前景加地面与损耗的 `d₀` 真值 `f_j`，`j` 为冻结的天空场景。对每个 `j` 用**同一预注册平滑前景族**拟合 `f_j`，得到 `f̂_j`、Jacobian `J_j`；此步不使用注入的 21 cm 信号。令 `W_jᵀW_j=C_j⁻¹` 为一年噪声白化，`Q_j=orth(W_j J_j)`，`P_j=I−Q_jQ_jᵀ`。定义

```text
u_j = P_j W_j [h ⊙ T₂₁]             有效信号模板
r_j = P_j W_j [f_j − f̂_j]          前景拟合后残差
I_j = u_jᵀ u_j                     固定模板振幅 A 的 Fisher 信息
b_j = u_jᵀ r_j / I_j               残差导致的振幅偏差诊断
S_j = sqrt[I_j / (1 + r_jᵀ r_j)]    风险调整后的信号对噪声/残差比
FOM = min_j S_j                     单值，越大越好；无量纲
```

`S_j` 是**设计评分**，不是频率学显著性或后验信噪比；分母把任何未建模前景残差按其白化总功率惩罚，保守地包含与模板平行及正交的部分。若 `I_j≈0`、beam 无效、前景拟合失败或 LST 泄漏超出预注册容限，则 `FOM=0` 并返回状态原因。主 API 返回一个标量，同时可选诊断给出 `I_j`、`b_j`、残差范数、地平线响应、`η_rad` 与数值误差。不得把这个评分解读为完整七参数 21 cm 信号恢复率；候选 shortlist 仍需完整 signal+null 注入、拟合与误检检查。

实现提供可选 `fit_spectrum=True`：在不影响上述 FOM 的前提下，额外对每个天空情景的无噪声注入数据 `f_j + h T₂₁` 联合拟合正温度前景和一个模板振幅 `A`。输出 `A_fit T₂₁`（本征谱）及 `h A_fit T₂₁`（天线谱）和联合拟合残差。它只在固定模板张成的一维谱族内估计振幅，不是自由谱形重建，也不是实测数据的反演。可选拟合失败时保留原 FOM，并在该情景的谱条目标记失败。

若所有候选的 `FOM` 被前景模型残差压到几乎相同的极小值，先诊断固定模型/天空输入是否有插值伪迹及是否需要**统一预注册**的 beam 校正不确定度模型，再改下一版协议；不能在单个候选上调模型去提高分数。

## 面向生成式搜索的计算路径

1. **协议预计算**：冻结 360-bin 访问数、天空 cube 和 fiducial 信号；为每个天空频率图构建 `(Npix/4π) map2alm(iter=0)` 的 quadrature alms。`Protocol.fingerprint` 对实际数组与配置作 SHA-256。
2. **候选预处理**：从全天 beam alms 计算效率和地面分割，应用天顶地平线遮罩并旋转一次到首个 LST bin 的赤道坐标系。保存 `sky_ref_alm` 后，同一候选反复评分无需变换。若生成器直接输出一致的 prepared alms，可跳过这一步。
3. **热循环**：按 packed `m` 分块求 `Σ_l conj(B_lm) S_lm`；以 `exp(i m ΔLST)` 一次合成 360-bin TOD。`d₀` 为 `m=0`，但噪声使用每个 bin 的 `Tsys` 与访问数，因此保留 beam 方位结构的时变影响。没有逐时刻球谐旋转，也不运行 MCMC。
4. **线性代数**：每候选每情景对约 66 个频点拟合一次固定阶数正温度前景，以 QR 白化投影后计算 `I`、残差与单值评分。旧的 zonal kernel 只用于角网格诊断，其 LST 均值噪声近似不作为主评分。
5. **慢速参考**：对入围候选，用同样遮罩后的本地 alm 调用 limTOD 逐 LST 旋转及 `mmode.solve(m_trunc=0)`，核对 `d₀`、噪声与 signal throughput；真实协议使用前还须做角分辨率收敛和 signal/null 注入。

## Package/API 和验收顺序

distribution 为 `horn-design-score`、import 为 `horn_design_score`；主要模块是 `modes.py`（球谐输入/预处理）、`harmonic.py`（快速 m-mode）、`protocol.py`（冻结配置）、`score.py`（单值指标）、`reference.py`（limTOD 对照）及 `cli.py`。核心接口 `score_beam(beam, protocol, prepared_scorer) -> ScoreResult`，其中 `ScoreResult.value` 是唯一排行榜指标。CLI 读一个候选 NPZ，输出带文件哈希的 JSON；批量搜索由调用者复用 Python `HarmonicScorer` 实例。

原方案列出的后续科学验收（尚未完成）：

1. 固定并版本化实验/sky/signal 文件、limTOD commit 与 beam 数据格式；选一个合规参考 horn、一个无色散 beam 和一个故意带色散的对照。
2. 实现全方向增益积分、地平线/损耗模型、缓存和快路径；先验收等温天空、方位旋转、频率恒定 beam、能量积分、`T₂₁` 传递系数等解析/变形检验。
3. 接上游 limTOD 的闭合 LST 求解交叉验证；对 8 h 弧专门证明有效性检查会拒绝严重泄漏，而不会给虚高分。
4. 实现前景投影和评分；用 signal/null 注入和一个非线性前景拟合复核 Fisher 排名、偏差诊断和失败状态。验证排序对积分分辨率、天空情景、通道平均与浮点精度稳定。
5. 在目标机器测量缓存构建时间、单候选/批量耗时和峰值内存，写入基准；只在验证等价后采用更激进的近似。

工程交付包括可安装 wheel、公开 API 与 CLI、合成可复现实例、球谐快慢路径数值比较及失败状态。真实协议输入锁定、nside 收敛、候选排序稳定性、完整注入测试和实际探测率属于后续科学验收。
