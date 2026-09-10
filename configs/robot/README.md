# Robot configurations

SO-101 的端口、相机和运行时安全限位将在真机接入阶段加入。训练数据使用的数据集级硬件档案由 `real_so101_vla_rl.data.build_robot_profile` 从实际 LeRobot 标定文件生成，并保存在数据集的 `project_meta/robot_profile.json` 中。

档案只记录训练数据的单位和来源，不能替代后续需要单独设置的真机安全限位。
