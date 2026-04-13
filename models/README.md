# Models 目录说明

`models/` 用于存放不同控制/学习方法的独立实现。

## 建议结构

```text
models/
  方法A/
    env.py
    train.py
    test.py
    README.md
  方法B/
    ...
```

## 约定

- 每个方法尽量自包含（训练、测试、配置在同目录）。
- 公共资源（URDF、mesh、数据）不要复制到 `models/`，统一引用项目根目录下的 `assets/`、`data/`。
- 新方法优先复用主工程中的工具模块，避免重复维护同名文件。
