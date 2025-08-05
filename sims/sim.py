import time

import mujoco
import mujoco.viewer
import numpy as np

XML_PATH = "models/dog/dog.xml"


def _get_state(m, d):
    """Gathers the state of the simulation into a dictionary."""
    state = {"time": d.time}

    # Gather geom information.
    geom_names = ["body", "leg_fl", "leg_fr", "leg_br", "leg_bl"]
    state["geoms"] = {}
    for name in geom_names:
        geom_id = m.geom(name).id
        body_id = m.geom(name).bodyid
        state["geoms"][name] = {
            "pos": d.geom_xpos[geom_id].copy(),
            "vel": d.cvel[body_id][:3].copy(),  # Linear velocity.
            "ang_vel": d.cvel[body_id][3:].copy(),  # Angular velocity.
        }

    # Gather joint information.
    joint_names = ["hip_fl", "hip_fr", "hip_br", "hip_bl"]
    state["joints"] = {}
    for name in joint_names:
        joint_id = m.joint(name).id
        qpos_adr = m.jnt_qposadr[joint_id]
        qvel_adr = m.jnt_dofadr[joint_id]
        state["joints"][name] = {
            "angle": d.qpos[qpos_adr],
            "velocity": d.qvel[qvel_adr],
        }

    return state


def run_simulation(control_func, duration=10.0, view=True):
    """
    Runs a simulation of the dog model.

    Args:
        control_func: A function that takes the state dictionary and current
            time, and returns a numpy array of 4 actuator controls.
        duration: The duration of the simulation in seconds.
        view: Whether to show the MuJoCo viewer.
    """
    m = mujoco.MjModel.from_xml_path(XML_PATH)
    d = mujoco.MjData(m)

    if view:
        with mujoco.viewer.launch_passive(m, d) as v:
            start_time = time.time()
            while v.is_running() and d.time < duration:
                step_start = time.time()
                state = _get_state(m, d)
                controls = control_func(state, d.time)
                d.ctrl[:] = controls
                mujoco.mj_step(m, d)
                v.sync()
                time.sleep(max(0.0, m.opt.timestep - (time.time() - step_start)))
    else:
        while d.time < duration:
            state = _get_state(m, d)
            controls = control_func(state, d.time)
            d.ctrl[:] = controls
            mujoco.mj_step(m, d)


if __name__ == "__main__":

    def passive_controller(state, time):
        """A simple controller that does nothing."""
        return np.zeros(4)
    
    def sinusoidal_controller(state, time):
        """A controller that applies a sinusoidal torque to the hip actuators."""
        return 45 * np.sin(np.pi * time)

    run_simulation(sinusoidal_controller, duration=100.0)
