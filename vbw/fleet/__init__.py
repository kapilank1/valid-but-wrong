"""Fleet mode: many vehicles' telemetry scored by a sharded set of detector workers (one Kubernetes StatefulSet).

    sharding.py  which worker owns which vehicle (rendezvous hashing)
    vehicle.py   one vehicle's stream: replay -> MITM attack -> CRC-framed link -> L0 -> detector
    worker.py    one shard: paces its vehicles in real time, reshards when the set of live workers changes
    metrics.py   Prometheus text exposition (/metrics) with no extra dependencies
    k8s.py       finds the Ready worker pods through the Kubernetes API

Run one worker locally:  uv run python -m vbw.fleet.worker   (FLEET_* env vars, see worker.py)
"""
