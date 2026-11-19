# camera-shake

## 题面

给爆炸加了镜头震动后，回放每次都不一样、镜头还会翻：trauma 叠加没有钳制，超过 1 后位移爆炸；衰减按线性而不是半衰期指数，半衰期不准确；噪声用了 random，每帧不同步、回放不可复现；位移直接按 trauma 线性映射，抖动幅度不自然；三轴共用同一个噪声，位移总是沿对角线；旋转单位用弧度却按度标注，实际抖动小 57 倍；world 空间位移不按相机朝向旋转，前后抖动变成左右；俯仰钳制不生效，镜头翻转；批量为每条事件重置 trauma，前一次震动被丢弃；批量遇非法事件只跳过，输出缺帧。shake/trauma.py 是核心，tests/test_trauma.py 15个红。要修的是：add_trauma 钳制到 0..1，decay 按半衰期指数衰减且不出现负值，noise_1d 使用显式 seed 的确定性值噪声，offset 与 rotation 用 trauma 平方映射且三轴独立、旋转以度为单位，apply 对 world 空间按相机朝向旋转位移，limit 钳制 pitch 与 roll 且不改变 yaw，shake_batch 顺序累积 trauma 并保证任一非法事件整体回滚。

## 运行方式

```bash
cd D:\gsb-20261119\camera-shake
python -m pytest tests -q
```

初始状态应恰好输出：

```text
15 failed
```

15 条测试全部因桩函数 `raise NotImplementedError` 而失败；收集、导入和语法检查必须通过，不能出现 `error`。实现后测试应全部通过。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 图形渲染 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## 测试约束

`tests/` 是验收测试，不得修改、删除或新增；修复只能落在 `shake/trauma.py`。所有 trauma 取值归一化到 `0..1`，旋转单位为度，断言允许 `1e-6` 容差；噪声必须使用显式传入的 seed，禁止使用全局随机状态。
