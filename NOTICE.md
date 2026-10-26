# 来源与授权说明

本项目采用 [MIT License](LICENSE)。应用实现与内置 SVG 图形为本项目原创；未复制下列参考项目的源码、模型权重或图片。系统字体通过用户设备加载，没有随项目分发字体文件。

## 参考资料

- [Transformer Explainer](https://github.com/poloclub/transformer-explainer)：交互解释器的体验参考。
- [Attention & Linear Mixer Visualizer](https://github.com/MatthewBonanni/attn-viz)：attention、张量形状与并行表达的功能比较参考。
- [Hugging Face Transformers cache explanation](https://huggingface.co/docs/transformers/cache_explanation)：因果缓存原理参考。
- [Qwen3-32B configuration](https://huggingface.co/Qwen/Qwen3-32B/blob/9216db5781bf21249d130ec9da846c4624c16137/config.json)：容量映射参数，固定版本并于 2026-10-26 核对；未包含或分发其模型权重。

测试与格式工具仅为开发依赖，其版本、来源与完整性记录在 `package-lock.json`。静态构建不打包这些依赖。新增第三方代码或素材时，需要维护其原有授权与出处。
