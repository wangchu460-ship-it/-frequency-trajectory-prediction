# G0_BALANCED e53：论文方法与代码定义（冻结版）

## 使用范围

本文档是当前论文主模型的唯一代码口径。正式模型为 **G0_BALANCED e53**：seed 789、系统平衡训练、以修正后的 POST_F 关系为输入。它是当前论文中的参考模型；当前 LOCAL、FCAR 双图和其他历史结构都不应混入其方法描述。

checkpoint 的 SHA256：`4c61eee32611eb8f63c5efbef92b2821482fd4c2ee54583ef88dcb3195d00a2a`。

## 术语表

| 规范术语 | 首次定义与论文中的含义 | 不应混用的旧名称 |
|---|---|---|
| G0_BALANCED | 采用系统平衡训练合同的四层物理关系图网络 | G0、CurrentFull、FCAR-Full |
| apparatus/device node | 显式的 SG、GFM 或 GFL 装置节点；频率轨迹的预测对象 | 母线节点 |
| bus node | 网络中的被动母线记录；不参与 F 关系传播 | device node |
| F relation | 故障后 POST_F 状态下设备—设备耦合关系，边权为归一化的 \(|F_{ij}|\) | 母线—线路拓扑消息 |
| F-message | 由源设备隐变量和 F 边属性构成的可学习消息聚合 | raw-neighbor LOCAL message |
| F-Laplacian | \(\sum_j w_{ij}(h_j-h_i)\) 的差分扩散项 | 旧 FCAR 的 bus-to-bus 传播 |
| HIGH device | 真值中心化频率偏差能量超过系统冻结阈值的设备 | 由预测值筛选的设备 |
| centered differential trajectory | 设备频率减去同场景、同时间、有效设备均值后的轨迹 | 绝对频率轨迹 |

## 1. 可直接写入论文的中文方法稿

### 3.1 问题定义与图表示

给定故障后的电力系统场景，目标是预测各显式装置节点 \(i\) 的频率偏差轨迹 \(\hat y_i(t)\)。预测对象仅包括同步发电机（SG）、构网型变流器（GFM）和跟网型变流器（GFL）。输入保留装置与母线节点的系统记录；但用于物理关系传播的 \(F\) 图仅定义在装置节点之间。对于场景 \(s\)，记 POST_F 关系为 \(\mathcal E_F^{(s)}\)，边 \((j,i)\in\mathcal E_F^{(s)}\) 的标量耦合权重为归一化后的 \(w_{ij}=|F_{ij}|\)。因此，模型以故障后的电气耦合状态而非故障前固定关系进行预测。

需要强调的是，G0 不包含 device\(\rightarrow\)bus\(\rightarrow\)bus\(\rightarrow\)device 的局部支路。母线节点不参与 \(F\)-message 或 \(F\)-Laplacian 的端点计算；当前模型也不在母线之间汇总或更新邻居消息。母线记录仅作为原始图输入的一部分，并通过末端的全图汇总间接进入轨迹解码。这一界限将当前 G0 与旧 FCAR 双图结构及已停止的 LOCAL 路径区分开来。

### 3.2 四层 F-message + F-Laplacian 编码器

节点原始特征首先经线性层、LayerNorm 和 SiLU 映射至 128 维初始表示 \(h_i^{(0)}\)。随后堆叠四个物理关系块。第 \(\ell\) 层中，F-message 分支先将六维 F 边属性编码为 \(e_{ij}^{(\ell)}\)，并以源节点状态及边表示形成消息：

\[
m_i^{(\ell)}=\operatorname{MLP}_{\mathrm{upd}}^{(\ell)}\!\left(
\left[h_i^{(\ell)},\;\sum_{j:(j,i)\in\mathcal E_F} w_{ij}\,
\operatorname{MLP}_{\mathrm{msg}}^{(\ell)}\!\left([h_j^{(\ell)},e_{ij}^{(\ell)}]\right)\right]\right).
\]

并行的 F-Laplacian 分支保留相对状态而非只传递邻居绝对表示：

\[
\ell_i^{(\ell)}=\operatorname{Proj}_{L}^{(\ell)}\!\left(
\sum_{j:(j,i)\in\mathcal E_F}w_{ij}\big(h_j^{(\ell)}-h_i^{(\ell)}\big)\right).
\]

两项物理残差在每层以可学习的 \(\eta_\ell\) 相加，随后经可学习物理门控、LayerNorm、SiLU 和 dropout 更新：

\[
r_i^{(\ell)}=m_i^{(\ell)}+\eta_\ell\ell_i^{(\ell)},\qquad
h_i^{(\ell+1)}=\operatorname{Dropout}\!\left[\operatorname{SiLU}\!\left(
\operatorname{LN}\big(h_i^{(\ell)}+\gamma_\ell r_i^{(\ell)}\big)\right)\right],
\]

其中 \(\gamma_\ell=\sigma(a_\ell)\)。\(F_{ii}\) 不作为自环输入；拉普拉斯项中的 \(h_j-h_i\) 已显式包含度项，避免重复计入对角信息。没有 F 边关联的节点在该层物理残差为零。

### 3.3 直接轨迹解码器

每个预测装置的解码 token 由其四层后的装置嵌入、整个输入图嵌入的 mean/max pooling，以及 11 个控制器数值特征和 3 个装置标志组成。数值控制器特征先经带符号 \(\log(1+|x|)\) 变换；标志对应 governor、GFM 与 GFL 身份。token 经两层 128 维 MLP 得到装置表示。

时间 \(t\) 相对故障时刻归一化为 \(\tau\in[0,1]\)，并编码为 \(\tau\)、\(\tau^2\)、\(\sqrt\tau\) 与四个频率 \(1,2,4,8\) 的正余弦特征，共 11 维。时间编码器生成 FiLM 参数 \((\Gamma(t),\beta(t))\)，调制装置 token 后，由 128\(\rightarrow\)128\(\rightarrow\)64\(\rightarrow\)1 的 MLP 在每个时间点直接输出 \(\hat y_i(t)\)。模型不通过递推状态方程或额外的参数校正头生成轨迹，也不在故障时刻施加硬零锚定。

### 3.4 面向空间异质性的训练目标

令 \(M_i(t)\) 表示有效时间掩码，\(\mathcal L_{\mathrm{OLD}}\) 表示冻结的原始 family-balanced 场景轨迹损失。为强调论文关心的设备相对频率偏离，先以每个场景、每个时刻的有效设备均值进行中心化：

\[
y_i^c(t)=y_i(t)-\frac{\sum_k M_k(t)y_k(t)}{\sum_k M_k(t)},\qquad
\hat y_i^c(t)=\hat y_i(t)-\frac{\sum_k M_k(t)\hat y_k(t)}{\sum_k M_k(t)}.
\]

HIGH 标签只由真值 \(y_i^c\) 的全时域梯形积分能量确定：

\[
i\in\mathrm{HIGH}\iff
\sqrt{\frac{\int M_i(t)[y_i^c(t)]^2\,dt}{\int M_i(t)\,dt}}\geq\theta_{\mathrm{system}},
\]

其中 \(\theta_{\mathrm{IEEE39}}=0.0184240117\) Hz，\(\theta_{\mathrm{NPCC140}}=0.0128750540\) Hz。辅助损失是在 HIGH 设备的中心化轨迹上计算 0–2 s 和 2–15 s 两个窗口的场景损失平均值 \(\mathcal L_{\mathrm{spatial}}\)。总的单场景损失为

\[
\mathcal L_{\mathrm{scene}}=\mathcal L_{\mathrm{OLD}}+1.7879867682\,\mathcal L_{\mathrm{spatial}}.
\]

该训练窗口与正式科学评价窗口不同：正式主终点固定为 HIGH centered differential RMSE 的 0–2 s 与 2–10 s，各系统等权；训练使用 2–15 s 辅助窗口以覆盖中后段演化。

### 3.5 跨系统平衡训练与 checkpoint 冻结

每个优化步严格抽取 IEEE39 的 4 个场景和 NPCC140 的 4 个场景，分别计算系统内平均损失后等权合并：

\[
\mathcal L=\tfrac12\operatorname{mean}_{s\in\mathrm{IEEE39}}\mathcal L_{\mathrm{scene}}(s)
+\tfrac12\operatorname{mean}_{s\in\mathrm{NPCC140}}\mathcal L_{\mathrm{scene}}(s).
\]

每个 epoch 完整覆盖两个系统各 3,500 个训练场景，共 875 个优化步；每轮均在冻结的 Validation1000 上评估。训练采用 AdamW（学习率 \(10^{-4}\)，weight decay \(10^{-4}\)）、梯度裁剪 1.0、FP32、无 scheduler，seed 为 789。模型从相同冻结初始化训练 60 epochs。整个选择和训练过程未读取 Test、OOD 或 WECC179 数据。

checkpoint 采用预先冻结的 V2 五轮 minimax 规则，而非单个平均分数或单轮最优值。四个 core endpoint（IEEE39 0–2、IEEE39 2–10、NPCC140 0–2、NPCC140 2–10 s）各自以 G0 epoch 1 的对应值归一化并等权得到 \(S_{\mathrm{HIGH}}(e)\)。对候选中心轮 \(e\)，在 \(\{e-2,\ldots,e+2\}\) 上依次最小化 \(U_5=\max S_{\mathrm{HIGH}}\)、窗口均值、中心轮值与 epoch。该规则选择 G0_BALANCED e53；单轮 raw minimum 在 e58，仅作趋势诊断而未作为正式模型。

## 2. 代码到论文模块的对应关系

| 论文模块 | 已冻结实现 | 代码事实 |
|---|---|---|
| 模型构造与 G0 消融开关 | `outputs/G0_LOCAL_TOPOLOGY_SERVER_V2/g0_runtime/run.py` | `FULL` 实例化 `F_MESSAGE_PLUS_LAPLACIAN`；`NO_LAPLACIAN` 只保留 F-message；`NO_MESSAGE` 只保留 \(\eta F\)-Laplacian。 |
| F-message + F-Laplacian 合并 | `outputs/G0_LOCAL_TOPOLOGY_SERVER_V2/g0_runtime/operator_screen.py` | 每个 block 返回 `message + eta_lap * laplacian`；\(\eta\) 为可学习标量，初始化为 0.05。 |
| 物理关系计算 | `work/reuse/UNIFIED_F_FCAR_FULL_V1_PACKAGE/formal/frozen/02_CONTROLLER_SCALE/runtime/backbone/haag_layers.py` | `node_type_id >= 0` 才可成为 F 边端点，母线 `node_type_id=-1` 被排除；第 4 个边属性列为归一化 \(|F_{ij}|\)。 |
| 四层编码器与直接解码器 | `work/reuse/UNIFIED_F_FCAR_FULL_V1_PACKAGE/formal/frozen/02_CONTROLLER_SCALE/runtime/backbone/models.py` | `hidden_dim=128`、`num_layers=4`、4 heads、dropout 0.05；输出键为 `direct_trajectory`。 |
| 损失与修正 POST_F 输入 | `outputs/G0_LOCAL_TOPOLOGY_SERVER_V2/g0_runtime/run.py` | `losses()` 按掩码和时间网格梯形积分计算 OLD 与 HIGH-centered spatial loss。 |
| 系统平衡合同 | `outputs/CROSS_SYSTEM_BALANCE_AND_EVIDENCE_V1/BALANCED_TRAINING_CONTRACT.json` | 严格 4+4 batch，系统外层等权，Train7000/Validation1000，POST_F。 |
| e53 选择证据 | `outputs/CROSS_SYSTEM_BALANCE_AND_EVIDENCE_V1/BALANCED_ANALYSIS/BALANCED_STABLE_SELECTION_V2.json` | raw best e58；冻结 V2 稳定选择 e53；训练边界已解析。 |

## 3. 论文中不能写成 G0 已经具备的内容

1. **不能写 G0 有母线—线路—设备局部消息路径。** 这是被独立测试后停止的 LOCAL 分支，而不是 G0 e53。
2. **不能写 G0 证明了真实母线邻居内容必要。** LOCAL 的 FULL/OFF/NO_B2B/B2B_SELF 审计没有通过预设机制门槛。
3. **不能将 `PhysicalOnly` 的代码工厂名写成 FCAR 或“只使用母线物理传播”。** 当前 G0 的物理关系是 POST_F 的设备—设备 \(F\) 关系。
4. **不能把 GLOBAL mean/max pooling 描述为邻居消息传递。** 它是置换不变的全图 readout，不产生逐边、逐邻居的更新。
5. **不能以单种子 e53 声称随机初始化下的普遍稳健性。** 当前主模型证据为冻结 seed789 的 Validation1000；多种子、Test、OOD 与 WECC179 仍属于后续验证。

## 4. 主张—证据—边界

| 论文可用主张 | 直接证据 | 可写边界 |
|---|---|---|
| 系统平衡训练应保留 | 平衡合同使整模型的训练梯度冲突审计从负余弦率 0.500 降至 0，且四个系统/窗口中 G0 相对旧合同有三项明确 Validation 改善 | 这是训练合同修正的证据，不是 LOCAL 结构有效性的证据。 |
| F-message 与 F-Laplacian 是 G0 的两个可分物理关系通道 | 代码中二者独立计算、独立消融；正式 A3/A4 消融矩阵正在按同一合同运行 | 在 A3/A4 结果完成前，不应写“两个通道均已显著提高最终性能”。 |
| G0 聚焦 HIGH 设备的相对频率轨迹 | HIGH 标签、中心化辅助损失和正式 0–2/2–10 s core endpoint 均与该问题一致 | HIGH 是由真值预定义的分析/训练标签；不等价于模型预测风险。 |
| 当前 LOCAL 不进入主模型 | LOCAL 与后续 RAW/SELF/DIFF 审计均未形成跨系统一致且真实邻居内容必要的证据 | NPCC140 中段的局部路径现象可作为限制或未来工作，不能扩展为 G0 的机制。 |

## 5. 可放在图 1 的模型图注

**G0_BALANCED architecture.** Each scenario is represented by apparatus and bus records, while the post-fault \(F\) relation is defined only between explicit SG, GFM and GFL apparatus nodes. Four stacked blocks combine a learned \(F\)-message residual with an \(F\)-Laplacian difference residual. Apparatus embeddings, graph mean/max summaries and controller descriptors are modulated by a time encoder and decoded directly into device-level frequency-deviation trajectories. No device-to-bus-to-bus-to-device local message branch is used in G0_BALANCED.

## 6. 英文 Methods 段落草稿

**Physics-relation trajectory predictor.** We predict the frequency-deviation trajectory \(\hat y_i(t)\) of each explicit synchronous-generator (SG), grid-forming-converter (GFM), and grid-following-converter (GFL) node after a disturbance. The input retains apparatus and bus records, whereas the post-fault physical relation \(\mathcal E_F\) is defined only between apparatus nodes. Its edge weight is the normalized coupling magnitude \(w_{ij}=|F_{ij}|\). Thus, buses are not endpoints of the \(F\)-message or \(F\)-Laplacian computations and G0 contains no device-to-bus-to-bus-to-device local branch.

Node features are mapped to a 128-dimensional latent state and processed by four physics-relation blocks. At layer \(\ell\), the message residual aggregates a learned function of the source state and six-dimensional edge descriptor, while the Laplacian residual aggregates the weighted state difference, \(\sum_j w_{ij}(h_j^{(\ell)}-h_i^{(\ell)})\). The two residuals are combined as \(r_i^{(\ell)}=m_i^{(\ell)}+\eta_\ell\ell_i^{(\ell)}\), where \(\eta_\ell\) is learned, and fused with the node state through a learned physical gate, LayerNorm, SiLU, and dropout. The resulting apparatus embedding is concatenated with graph mean/max summaries and device-controller descriptors. A time encoder supplies FiLM parameters from normalized post-event time and a direct decoder outputs the trajectory at every sampled time point.

We train with the frozen original family-balanced trajectory loss augmented by a spatial objective for devices with high true centered deviation energy. Centering subtracts the mean frequency across valid devices at each time. The spatial term averages masked centered-trajectory losses over 0–2 s and 2–15 s and is weighted by 1.7879867682. To balance the IEEE39 and NPCC140 systems, each optimization step contains four scenarios from each system and optimizes the equally weighted mean of their within-system scene losses. The final G0 checkpoint was selected on Validation1000 using a prespecified five-epoch minimax stability rule over equally weighted HIGH centered-differential RMSE endpoints for both systems and the 0–2 s and 2–10 s windows; no Test, OOD, or WECC179 data were used for training or selection.

## 7. 写作前仍需补入的项目

- 图中明确绘制：装置节点、被动母线记录、设备—设备 POST_F 边、四层物理块、全图 pooling、控制器 token、时间调制与 device trajectory decoder。
- 正式 Results 的 F-only、L-only、无物理关系和 PRE-event 结果必须来自正在执行的正式消融矩阵；不要用旧 30-epoch 或 FCAR 消融代替。
- 外部基线、multi-seed 和 WECC179 的结果完成后，再写“优于既有方法”或跨系统泛化结论。
