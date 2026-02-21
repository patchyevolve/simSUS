from simulation import Simulation


def test_clear_trails_removes_existing_points():
    sim = Simulation()
    sim.load_preset("solar_system")

    for body in sim.bodies:
        body.record_trail()

    assert any(body.trail for body in sim.bodies)

    sim.clear_trails()

    assert all(len(body.trail) == 0 for body in sim.bodies)
