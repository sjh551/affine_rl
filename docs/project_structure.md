# Project Structure Notes

## 本次整理内容

1. 将小车模型目录从项目根迁移到：`assets/ranger_mini_v3/`
2. 修复以下文件的 URDF 路径为绝对路径解析：
   - `pybullet_env.py`
   - `control_ranger_mini_test.py`
   - `models/DWA/DWA_env.py`
   - `models/uniform_scale_transform/uniform_scale_env.py`
   - `old_files/main_demo.py`

## 第二阶段整理内容

1. 新增统一路径模块：`project_paths.py`
2. 训练产物统一输出到 `outputs/`（模型、checkpoint、tensorboard、图像、autoencoder权重）
3. 统一入口脚本放入 `scripts/`
4. 更新并兼容以下脚本的模型/产物路径：
   - `train.py`
   - `test.py`
   - `evaluate_success_rate.py`
   - `train_autoencoder.py`
   - `data/collect_data.py`
   - `speed_test.py`
   - `models/uniform_scale_transform/train.py`
   - `models/uniform_scale_transform/test.py`
5. 已将历史产物迁移到 `outputs/` 对应子目录

## 兼容性说明

- 现在从任意工作目录启动上述脚本，都可以正确加载 URDF。
- `models/` 作为算法方法集合的定位已固定，后续仅扩展方法代码即可。
- 资源文件统一在 `assets/`，避免散落在根目录。
