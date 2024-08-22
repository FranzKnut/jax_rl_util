import jax


def render_brax(env, states, render_steps=100, render_start=0, camera="track"):
    from brax.io import image

    steps = len(states.pipeline_state.q)
    states_to_render = [
        jax.tree.map(lambda x: x[n], states.pipeline_state)
        for n in range(steps)
        if n > render_start and n < render_start + render_steps
    ]
    return image.render_array(env.sys, states_to_render, camera=camera)
