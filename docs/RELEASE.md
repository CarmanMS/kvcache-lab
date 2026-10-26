# v0.1.0 发布检查清单

本清单用于后续版本发布。v0.1.0 的实际提交、CI 与部署记录见文末及 GitHub Release。配置 CI 或生成构建目录不等于完成发布。

源码仓库：https://github.com/CarmanMS/kvcache-lab 。

## 1. 固定候选版本

- [ ] 确认发布提交，记录完整 commit SHA；后续所有检查和产物必须对应它。
- [ ] `package.json` 与锁文件版本均为 `0.1.0`，package 许可证为 MIT。
- [ ] [CHANGELOG.md](../CHANGELOG.md)描述实际功能、修复和已知边界。
- [ ] 仓库和产物随附 `LICENSE`、`NOTICE.md`，第三方内容来源可追溯。
- [ ] 没有把本机配置、密钥、私人数据、`tmp/` 或 `node_modules/` 纳入提交。

在源码目录执行这些只读核查命令：

```sh
git status --short
git diff --check
git rev-parse HEAD
node -e "const p=require('./package.json'); const l=require('./package-lock.json'); if(p.version!=='0.1.0'||l.version!==p.version||p.license!=='MIT') process.exit(1); console.log(p.version,p.license)"
```

`git status` 必须没有未纳入候选提交的产品改动；`git rev-parse HEAD` 失败意味着尚未建立可追溯的候选提交。

## 2. 本地检查与构建

```sh
npm ci
npm run check
npm test
npm run build
node -e "const fs=require('node:fs'); for (const f of ['dist/index.html','dist/LICENSE','dist/NOTICE.md']) fs.accessSync(f); console.log('Required release files exist')"
```

- [ ] 所有命令退出码为 0；记录 Node/npm 版本、系统和执行日期。
- [ ] 检查产物只包含预期的站点资源和分发声明，源码中已删除的资源不再存在。
- [ ] 依赖检查没有新增的、未经评估的严重漏洞；保存检查时间，不能把“当时无公告”视为长期保证。

## 3. 实际跨平台 CI

`.github/workflows/ci.yml` 配置了六组检查：Linux、Windows、macOS 各运行 Node.js 22 和 24。远端需要包含候选提交、锁文件及工作流，并启用 GitHub Actions。

仓库建立后，可用以下命令查看证据：

```sh
gh run list --repo CarmanMS/kvcache-lab --workflow ci.yml --limit 10 --json databaseId,headSha,status,conclusion,url
gh run view --repo CarmanMS/kvcache-lab --json headSha,status,conclusion,jobs,url
```

第二条命令按提示选择对应的运行。若当前没有对应运行，可以由有权限的维护者在 Actions 页面手动触发 `CI`，选择候选提交所在分支。

- [ ] 运行的 `headSha` 等于候选提交 SHA。
- [ ] 六组 job 都执行了 `npm ci`、`npm run check`、`npm test`、`npm run build`，最终结果为 `success`。
- [ ] 记录运行链接；缺失、跳过、取消或只执行部分平台不算通过。
- [ ] 若 CI 失败并修复，使用新提交重新执行，不能复用旧提交的绿色结果。

## 4. 浏览器验收

```sh
npm run preview -- --port 5175
```

- [ ] 桌面与 390px 页面正常，控制台没有未处理错误。
- [ ] 4 层 MHA、8 个 Q heads、16 token 组合无页面横向溢出，图内可滚动，选择对象后可见。
- [ ] 草稿、展开状态、光标和输入焦点在播放、提交、提示消失和视图切换时保持。
- [ ] Prefill、decode、撤销、TP 切换、请求结束与淘汰行为符合界面说明。
- [ ] 键盘操作、焦点可见性和减少动画偏好可用；记录浏览器版本与截图。

## 5. 公开发布

- [ ] 更新[安全说明](../SECURITY.md)为确实可用的私下报告方式；公开仓库启用私密漏洞报告时，填入真实入口。
- [ ] 确认仓库可见性、项目介绍、源码地址和许可证展示正确。
- [ ] 让 `v0.1.0` 标签指向已通过上述检查的提交；发布说明使用 [CHANGELOG.md](../CHANGELOG.md)中对应版本内容。
- [ ] 将版本记录的“待发布”改为实际发布日期，并记录源码版本、CI 链接和部署地址。
- [ ] 按[静态部署说明](DEPLOYMENT.md)部署同一版本，完成 HTTPS、子目录资源路径和发布后操作核验。

标签、仓库可见性、Release 和站点部署会改变远端状态，应由维护者在确定发布范围后执行。本清单本身不会执行这些操作。

## 发布证据

| 证据                  | 记录                                                                                                           |
| --------------------- | -------------------------------------------------------------------------------------------------------------- |
| 候选提交 SHA          | 以 v0.1.0 标签指向的提交为准；完整 SHA 记录于 Release                                                          |
| 六组 CI 运行链接      | [首轮全平台通过](https://github.com/CarmanMS/kvcache-lab/actions/runs/40289789257)；发布提交复验链接见 Release |
| 浏览器验收环境与结果  | macOS Chrome154，桌面与390px、完整操作链通过                                                                   |
| `v0.1.0` 标签/Release | [v0.1.0](https://github.com/CarmanMS/kvcache-lab/releases/tag/v0.1.0)                                          |
| 站点地址与发布后验收  | https://carmanms.github.io/kvcache-lab/ ，具体记录见 Release                                                   |
| 实际发布日期          | 2026-10-26                                                                                                     |
