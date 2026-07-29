# v0.6.0 发布证据

结论：`0.6.0` 的 Rust crate 与 Python wheel 已发布；当前 `origin/main` 的全部更新尚未发布。

不可变证据链（2026-07-29 核验）：

- 远端注释标签 `v0.6.0` 的对象是 `db0566fc28977343dee33b68d3ceb1f18f464e5a`，peeled commit 为 `565e5a98f91d760afa41fc4846f7657257136082`。
- 远端注释标签 `crate-v0.6.0` 的对象是 `49da5ad34a5070a85df681d85fd10d6af7b87527`，peeled commit 同为 `565e5a98f91d760afa41fc4846f7657257136082`。
- GitHub Actions 在该 SHA 成功：Release run `29596883929`、Release Crate run `29596491200`。
- PyPI 的 `wbt==0.6.0` 未被 yank；macOS arm64 wheel SHA-256 为 `d60c8d1a7ef1b83654b28b4ea2791594a5f3a52e626bddc739f310385e8cec60`。
- crates.io 的 `wbt==0.6.0` 未被 yank；crate checksum 为 `4f6781b2e914ee683b98a12bd8bd2c2c086663de74c9f04b2a894c9070cd4069`。

`v0.6.0..origin/main` 的完整提交清单：

```text
d903ed2129d8df171f71e1d37c7414859cd12145 feat: 增加波动率归一空头超额曲线
42d13f4846d4f4da2b1583c63a0f1347d9614820 fix: 优化累计收益曲线图例名称
555ee6fc3048d53c575d4cbaa015d87bfca9b17e Merge pull request #21 from zengbin93/worktree-skz-283-plan
```

因此，不能将可移动的 `main` 或 GitHub Release target 当作发布源码证据，也不能声称当前 main 已全部发布。
