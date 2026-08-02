# 贡献指南

欢迎参与 pixbox 的开发与构建。这里的重点是帮助开发者尽快完成本地环境准备、理解项目结构、执行质量检查、构建产物，并顺利提交贡献。

## 1. 项目结构

建议先了解三个核心部分：

- src/pixbox/core：像素格式、颜色范围、颜色传输函数、颜色主以及颜色空间转换逻辑
- src/pixbox/gui：基于 PySide6 的桌面界面，负责文件打开、设置面板、播放控制与显示
- tests：用于覆盖 core 与 gui 的行为，确保修改后稳定可回归

此外，仓库里还有这些关键文件：

- noxfile.py：定义统一的开发构建与质量检查流程
- .pre-commit-config.yaml：定义提交前和推送前要执行的钩子
- pixbox.spec：PyInstaller 的打包配置，用于构建可执行文件

## 2. 开发环境准备

### 2.1 安装依赖

请先确保已经安装了 Python 3.12 或更高版本。推荐使用 uv：

```bash
uv sync --group dev
```

这一步会安装项目运行依赖和开发依赖，包括测试、格式化、类型检查以及 pre-commit 工具。

### 2.2 安装 pre-commit 钩子

```bash
uv run pre-commit install
```

安装完成后，提交前会触发 pre-commit 检查；同时，仓库还配置了 pre-push 阶段的钩子，因此在执行 git push 时也可能触发额外检查。

## 3. 推荐开发工作流

建议按下面顺序进行：

1. 明确问题或需求，并尽量先在本地复现。
2. 先修复/实现功能，再补充或更新测试。
3. 运行本地检查。
4. 确认通过后再提交。

## 4. 代码风格与质量检查

### 4.1 运行测试

```bash
uv run pytest tests
```

### 4.2 运行 Ruff 检查与格式化

```bash
uv run ruff check .
uv run ruff format .
```

### 4.3 运行 mypy 类型检查

```bash
uv run mypy . --strict
```

### 4.4 使用 nox 统一执行

一次性执行项目里的常见质量检查，可以使用：

```bash
uv run nox
```

nox 会把 lint、格式检查、类型检查和测试串起来，适合在准备提交前使用。

## 5. pre-commit 与 pre-push 的含义

当前仓库配置了多个 pre-commit 钩子，主要包括：

- 文件完整性检查：空白字符、换行、JSON/TOML/YAML/XML 格式
- 代码风格检查：Ruff 检查与格式化
- 类型检查：mypy
- 语法升级检查：pyupgrade
- 拼写检查：codespell
- 提交信息规范检查：commitizen

其中 commitizen 的 hooks 还包含 pre-push 阶段，这意味着执行 git push 时也可能被拦住。如果提交信息或分支状态不符合配置要求，push 会失败。

## 6. 提交规范

如果手写提交信息，建议遵循类似下面的风格：

```text
feat: add support for new pixel format
fix: correct yuv420 conversion path
refactor: simplify color transfer selection logic
```

提交时尽量做到：

- 一个提交只解决一个主题
- 提交信息要清楚表达改动内容
- 如果改动较大，建议先在 issue 或讨论中说明思路

## 7. 构建与打包

### 7.1 构建 Python 包

如果想生成分发包，可以使用：

```bash
uv build
```

### 7.2 构建可执行文件

仓库中已经提供了 PyInstaller 配置文件 [pixbox.spec](pixbox.spec)，适合用于生成桌面可执行文件。

在开发环境中，如果 PyInstaller 可用，可以基于该 spec 文件进行打包，例如：

```bash
pyinstaller pixbox.spec
```

打包产物通常会生成到 dist 目录下，便于做本地验证或发布使用。

## 8. 提交前 checklist

在真正提交前，建议至少确认：

- 相关测试已通过
- Ruff 检查与格式化已执行
- mypy 没有新的报错
- pre-commit 钩子没有报错
- 如果修改了功能逻辑，测试也已同步补齐

## 9. Pull Request 建议

提交 PR 时建议：

- 在说明中清楚写出改动目的与背景
- 说明是否修复了问题或新增了功能
- 说明测试与检查是否已执行
- 如果是较大改动，尽量附上简短的变更说明或示例

## 10. 许可证说明

本项目基于 GNU General Public License v3.0（GPL-3.0）发布。任何贡献都应被视为你同意在相同许可证下提供你的修改内容，且衍生作品也需保持 GPL 兼容的开源许可要求。
