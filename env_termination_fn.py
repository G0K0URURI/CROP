import numpy as np


def hopper_termination_fn(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape)
    if len(obs.shape) == 1:
        height = next_obs[0]
        angle = next_obs[1]
        not_done = np.isfinite(next_obs).all(axis=-1) \
                   * np.abs(next_obs[1:] < 100).all(axis=-1) \
                   * (height > .7) \
                   * (np.abs(angle) < .2)

        done = ~not_done
        # done = done[None]
        return done
    elif len(obs.shape) == 2:
        height = next_obs[:, 0]
        angle = next_obs[:, 1]
        not_done = np.isfinite(next_obs).all(axis=-1) \
                   * np.abs(next_obs[:, 1:] < 100).all(axis=-1) \
                   * (height > .7) \
                   * (np.abs(angle) < .2)

        done = ~not_done
        done = done[:]
        return done


def walker2d_termination_fn(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape)
    if len(obs.shape) == 1:
        height = next_obs[0]
        angle = next_obs[1]
        not_done = (height > 0.8) \
                   * (height < 2.0) \
                   * (angle > -1.0) \
                   * (angle < 1.0)
        done = ~not_done
        # done = done[None]
        return done
    elif len(obs.shape) == 2:
        height = next_obs[:, 0]
        angle = next_obs[:, 1]
        not_done = (height > 0.8) \
                   * (height < 2.0) \
                   * (angle > -1.0) \
                   * (angle < 1.0)
        done = ~not_done
        done = done[:]
        return done

def halfcheetah_termination_fn(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    done = np.array([False]).repeat(len(obs))
    done = done[:]
    return done


termination_fn = {'hopper': hopper_termination_fn, 'walker2d': walker2d_termination_fn,
                  'halfcheetah': halfcheetah_termination_fn}
