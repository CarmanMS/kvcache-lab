# 独立安装与运行

KVCache Lab 可以作为独立仓库运行，不需要其他主仓、知识库、Python、GPU、模型权重、后端服务或 API Key。

## 准备环境

- 使用 Node.js 22 或 24；Node 安装包自带 npm。先运行 `node --version` 和 `npm --version` 确认命令可用。
- macOS、Windows 和 Linux 使用下面相同的 npm 命令。Windows 可在 PowerShell 或终端中执行。
- 使用支持 ES modules 的现代浏览器。真实布局和辅助技术的兼容性仍需按目标浏览器验收。

## 取得源码

v0.1.0 当前在[私有仓库](https://github.com/CarmanMS/kvcache-lab)验证。取得仓库访问权限后可以克隆；也可以解压维护者提供的完整源码包。

```sh
git clone https://github.com/CarmanMS/kvcache-lab.git
cd kvcache-lab
```

私有验证期间需要维护者授予访问权限。源码包必须包含 `package.json`、`package-lock.json`、`scripts/`、`src/`、`tests/` 和许可证，不应只复制单个 HTML 文件。

## 安装开发工具并启动

在含 `package.json` 的目录执行：

```sh
npm ci
npm run dev
```

打开 http://127.0.0.1:5174。开发服务器只监听本机；按 `Ctrl+C` 停止。

端口被占用时换一个端口，不需要结束其他项目的服务：

```sh
npm run dev -- --port 5175
```

`npm ci` 使用锁文件安装测试与格式化工具。浏览器运行产物没有第三方运行时依赖；首次安装仍需要访问 npm registry。

## 验证和预览构建

```sh
npm run check
npm test
npm run build
npm run preview -- --port 5175
```

打开 http://127.0.0.1:5175。开发与预览使用相同页面行为；预览读取 `dist/`，修改源码后需要重新构建。

构建过程会清理旧 `dist/`，不要在其中保存手工修改。临时验证文件统一放在 `tmp/`。

## 常见问题

| 现象                                           | 处理                                                                              |
| ---------------------------------------------- | --------------------------------------------------------------------------------- |
| 找不到 `node` 或 `npm`                         | 安装 Node.js 22/24，重新打开终端确认 PATH。                                       |
| `npm ci` 报锁文件与 package 不一致             | 确认源码来自同一提交；不要删除锁文件绕过检查。                                    |
| 端口占用                                       | 使用 `-- --port 5175` 或其他空闲端口。                                            |
| 预览提示找不到目录                             | 先执行 `npm run build`。                                                          |
| 直接双击 HTML 后页面空白                       | 使用 HTTP 服务器访问；`file://` 下的模块加载受浏览器限制。                        |
| 通过局域网地址或自定义域名访问本地服务器被拒绝 | 本地服务器只允许回环地址；需要分享时按[静态部署说明](DEPLOYMENT.md)发布 `dist/`。 |
| 修改源码后预览没有更新                         | 重新构建，刷新页面；确认浏览器打开的是预览端口。                                  |

贡献流程见 [CONTRIBUTING.md](../CONTRIBUTING.md)，计算假设见 [ARCHITECTURE.md](ARCHITECTURE.md)。
