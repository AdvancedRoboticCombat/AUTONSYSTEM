# nemesis_policy_engine

`saw_policy_node` flies a hammer-saw checkpoint from `../auton_training`
(`SAW_*.pt`, 44 inputs, 4 outputs) on the robot and in Gazebo.

- In: `/nemesis/centroid`, `/opponent/centroid` (camera positions), `/nemesis/imu` (required), `/nemesis/override`.
- Out: `/control/channels` = `[left tread, right tread, blade, arm]`. Blade and arm are held off unless `weapon_enabled` is set. Nothing is published while override is on.
- Settings (arena frame, robot sizes, arm timing, the opponent's card): `config/saw_policy.yaml`.
- The observation, arm gate and tread mixer are `../auton_training/deploy.py`.

Build with `scripts/build.sh` from the repository root; run with `scripts/all.sh`,
`scripts/autonomy.sh` or `scripts/sim.sh`.
