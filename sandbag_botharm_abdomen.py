import numpy as np
from gym_upperbody.envs import mujoco_env
from gym import utils


def mass_center(model):
    mass = model.body_mass
    xpos = model.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))[0]


class SandbagBotharmAbdomenEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'sandbag_botharm_abdomen.xml', 5)
        utils.EzPickle.__init__(self)


    def _get_obs(self):
        data = self.model.data

        right_shoulder_y_angle = self.model.data.qpos.flat[24:25]
        right_shoulder_x_angle = self.model.data.qpos.flat[25:26]
        right_shoulder_z_angle = self.model.data.qpos.flat[26:27]
        right_elbow_y_angle = self.model.data.qpos.flat[27:28]

        left_shoulder_y_angle = self.model.data.qpos.flat[28:29]
        left_shoulder_x_angle = self.model.data.qpos.flat[29:30]
        left_shoulder_z_angle = self.model.data.qpos.flat[30:31]
        left_elbow_y_angle = self.model.data.qpos.flat[31:32]

        right_shoulder_y_velocity = self.model.data.qvel.flat[23:24]
        right_shoulder_x_velocity = self.model.data.qvel.flat[24:25]
        right_shoulder_z_velocity = self.model.data.qvel.flat[25:26]
        right_elbow_y_velocity = self.model.data.qvel.flat[26:27]

        left_shoulder_y_velocity = self.model.data.qpos.flat[27:28]
        left_shoulder_x_velocity = self.model.data.qpos.flat[28:29]
        left_shoulder_z_velocity = self.model.data.qpos.flat[29:30]
        left_elbow_y_velocity = self.model.data.qpos.flat[30:31]

        abdomen_z_angle = self.model.data.qpos.flat[10:11]
        abdomen_y_angle = self.model.data.qpos.flat[11:12]
        abdomen_x_angle = self.model.data.qpos.flat[12:13]

        abdomen_z_velocity = self.model.data.qvel.flat[9:10]
        abdomen_y_velocity = self.model.data.qvel.flat[10:11]
        abdomen_x_velocity = self.model.data.qvel.flat[11:12]

        sandbag_velocity = self.model.data.qvel.flat[32:33]

        sandbag_pos = self.get_body_com("sandbag")

        target_pos = self.get_body_com("target")

        fist_target_distance = np.linalg.norm(self.get_body_com("fingertip_right")-self.get_body_com("target"))

        shoulder_pos = self.get_body_com("left_upper_arm")

        elbow_pos = self.get_body_com("left_lower_arm")

        theta = data.qpos.flat[17:20]
        return np.concatenate([
            data.qpos.flat[:3],
            self.get_body_com("fingertip_right")-self.get_body_com("target"),
        ])


    def _step(self, a):
        print(self.model.data.qvel.flat[0:])
        vec = self.get_body_com("fingertip_right")-self.get_body_com("target")
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = 0
        reward = reward_dist + reward_ctrl
        self.do_simulation(a, self.frame_skip)
        done = False
        if np.linalg.norm(vec) < 0.1:
            done = True
        return self._get_obs(), reward, done, dict(qpos=self.model.data.qpos.flat)


    def reset_model(self):
        max_x =  0.68
        min_x =  0.48
        max_y = -0.16
        min_y =  0.16
        max_z =  1.65
        min_z =  1.35
        qpos = self.init_qpos
        qvel = self.init_qvel

        while True:
            self.goal = [self.np_random.uniform(low=min_x, high=max_x, size=1),
                         self.np_random.uniform(low=min_y, high=max_y, size=1),
                         self.np_random.uniform(low=min_z, high=max_z, size=1),]
            if np.linalg.norm(self.goal) < 2:
                break
        qpos[:3] = self.goal

        self.set_state(qpos, qvel)

        return self._get_obs()


    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 4
        self.viewer.cam.distance = self.model.stat.extent * 1.8
        self.viewer.cam.lookat[2] += .61
        self.viewer.cam.elevation = -10
