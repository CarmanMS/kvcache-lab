# KVCache Lab

面向课堂讲授与自主探索的浏览器交互实验室：从 token、layer 和 Q/KV head，一直追踪到缓存页、GPU 和机器。教师可以自行组织教学问题与讲授顺序。

## 本地运行

需要 Node.js 22 或更新版本。macOS、Windows 和 Linux 使用相同命令。浏览器运行时无第三方依赖；测试与格式检查使用锁定的开发依赖。

```sh
npm ci
npm run dev
```

打开 http://127.0.0.1:5174。改端口可用 `npm run dev -- --port 5175`。

```sh
npm run check
npm test
npm run build
npm run preview
```

`dist/` 是完整静态站点。开发服务器只绑定本机，不启动模型服务。

## 可以做什么

- 在 MHA / GQA / MQA 之间切换，展开 2–4 层、4 或 8 个 Q heads，直接选择各层、各 head 和 token。
- 运行 prefill，逐 token decode，播放或撤销运行状态，编辑输入序列。
- 检查真实计算出的 Q/K/V、缩放点积、softmax 权重与加权 Value；历史 Query 是调试快照，不计入 KV 池。
- 查看 TP=1/2/4/8（按 Q head 数约束）下每张 GPU 的 head 归属、KV 副本、输出部分和以及机器分组。
- 比较公共前缀与相同后缀，切换模型身份，结束请求、保留缓存、淘汰无引用页并撤销。
- 从小模型映射到 Qwen3-32B 的真实维度，估算上下文长度、BF16/FP8 与 TP 对逻辑 KV 大小的影响。

## 教师使用示例（文档参考）

以下问题用于说明工具的教学能力，页面不预设问答入口或课程顺序。

1. 为什么不缓存 Q？查看 Query 和缓存池；单步后新增 Query 读取相同的历史 K/V。
2. 每层的 KV 一样吗？保持 token 不变，点选不同 layer，比较向量。
3. 相同文字能共享吗？在「请求与缓存」选择「相同后缀」，查看共享为零及深层 K 的差值。

## 计算与模拟边界

- 数值来自多层、确定性随机权重的教学 Transformer，并非训练好的 Qwen；续写 token 是预设输入，不宣称生成语义质量。
- 包含 pre-norm、Q/K head norm、RoPE、因果 attention、output projection、残差与 SwiGLU FFN。使用 JS 浮点，不执行 BF16/FP8 量化；精度选择仅改变容量估算。
- 缓存数值计算与物理页模拟分开验证。为查看任意历史快照，浏览器会重建教学数值；界面不把重建耗时当作真实推理性能。
- 页池按完整前缀页去重，模型身份隔离；只在页边界、已知完整页时匹配已有页。已经分配的部分页补齐后保持原槽位，不再合并迁移。它不是 SGLang 某个版本的逐行复现，没有硬件性能预测。
- 机器/GPU 是部署模拟；TP 输出部分和在数学上真实归并，无实际跨 GPU 通信。
- Qwen3-32B 容量来源是公开 config，32,768 token / BF16 的逻辑 KV 为 8 GiB；未计元数据、页对齐、量化 scale、权重与工作区。
- MoE、MLA、KDA 与更多并行方式列为后续阶段，首版不会把未实现的选项展示成可用功能。

## 代码

`src/math.js`：纯数学算子；`src/model.js`：增量与独立全前缀执行；`src/cache.js`：页池、TP 与容量；`src/controller.js`：实验状态；`src/views.js`：视图；`src/app.js`：事件与 DOM 生命周期；`src/styles.css`：响应式界面。

验证覆盖数值等价、因果前缀不变、分层差异、TP 归并、地址稳定性、引用回收与检查点恢复。

## 来源与开发记录

- 体验参考：https://poloclub.github.io/transformer-explainer/
- Attention 细节比较参考：https://github.com/MatthewBonanni/attn-viz
- Qwen 配置：https://huggingface.co/Qwen/Qwen3-32B/blob/9216db5781bf21249d130ec9da846c4624c16137/config.json
- KV 原理：https://huggingface.co/docs/transformers/cache_explanation
- 本仓库为原创实现，没有复制上述项目源码。

## 版本与贡献

当前为经过审查的早期 v0.1.0，正在[私有仓库](https://github.com/CarmanMS/kvcache-lab)验证，尚未公开发布。跨平台 CI 已配置，当前实际验证环境为 macOS；MoE、MLA、KDA 尚未实现。

- [架构与计算边界](docs/ARCHITECTURE.md)
- [开源前审查与验证](docs/REVIEW.md)
- [v0.1.0 版本说明](CHANGELOG.md)
- [独立安装](docs/INSTALL.md) · [静态部署](docs/DEPLOYMENT.md) · [发布检查清单](docs/RELEASE.md)
- [贡献指南](CONTRIBUTING.md) · [安全说明](SECURITY.md)
- [MIT 许可证](LICENSE) · [来源声明](NOTICE.md)
