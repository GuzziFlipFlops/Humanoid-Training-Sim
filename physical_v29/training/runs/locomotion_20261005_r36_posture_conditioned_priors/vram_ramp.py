"""Conservative, training-only VRAM evidence for checkpoint-boundary env ramps."""
import math

LADDER = (512, 1024, 2048, 3072, 4096, 6144, 8192)


class VramRamp:
    def __init__(self, reserve_mb=300, hold_seconds=180, mib_per_env=.5):
        self.reserve_mb = reserve_mb
        self.hold_seconds = hold_seconds
        self.mib_per_env = mib_per_env
        self.since = None
        self.last = None
        self.count = None
        self.ready = None
        self.cooldown_until = 0.
        self.last_status = {}

    def next_count(self, count):
        return next((n for n in LADDER if n > count), None)

    def observe(self, now, count, free_mb, training_ready=True, host_available_mb=None):
        target = self.next_count(count)
        needed = self.reserve_mb + math.ceil((target-count)*self.mib_per_env) if target else None
        # Native Isaac host allocations grow independently of PyTorch/VRAM.
        # Require host room too, rather than interpreting host failures as a
        # larger GPU allocation per added environment.
        host_needed = 2048+1024+math.ceil((target-count)*.75) if target else None
        host_healthy = host_available_mb is None or (math.isfinite(host_available_mb)
                       and host_available_mb >= (host_needed or 2048))
        enough = (target is not None and training_ready and free_mb is not None
                  and math.isfinite(free_mb) and free_mb >= needed and host_healthy
                  and now >= self.cooldown_until)
        continuous = self.last is not None and 0 <= now-self.last <= 15 and self.count == count
        if not enough:
            self.since = None
            self.ready = None
        elif self.since is None or not continuous:
            self.since = now
            self.ready = None
        elif now-self.since >= self.hold_seconds:
            self.ready = target
        self.last, self.count = now, count
        self.last_status = dict(current_envs=count, next_envs=target, required_free_mb=needed,
                                free_mb=free_mb, qualifying_seconds=0 if self.since is None else now-self.since,
                                hold_seconds=self.hold_seconds, ready_envs=self.ready,
                                mib_per_added_env=self.mib_per_env, cooldown_until=self.cooldown_until,
                                host_available_mb=host_available_mb,required_host_available_mb=host_needed,
                                host_healthy=host_healthy,
                                scope='Only live PPO allocation counts; eval/startup memory never qualifies')
        return self.last_status

    def consume(self):
        target = self.ready
        self.since = self.last = self.ready = None
        return target

    def memory_failure(self, now, resource='vram'):
        assert resource in ('vram','host')
        self.consume()
        self.cooldown_until = now+self.hold_seconds
        if resource=='vram':self.mib_per_env *= 1.25


def estimate_increment(benchmarks):
    pairs = sorted(benchmarks, key=lambda b:b['environments'])
    estimates = [(a['minimum_free_mb']-b['minimum_free_mb'])/(b['environments']-a['environments'])
                 for a,b in zip(pairs,pairs[1:]) if b['environments'] > a['environments']]
    # 0.5 MiB/env is above the observed 0.357 MiB/env on this host.
    return max([.5]+[v*1.25 for v in estimates if v > 0])


def training_iterations(minimum, step_seconds, count, hold_seconds=180):
    if count >= LADDER[-1]:return minimum
    # Leave time for warmup/measurement. Short blocks must not make a 180s
    # live-allocation requirement impossible on a fast GPU.
    measured=max(.5,step_seconds if step_seconds is not None else 3.5)
    return max(minimum,math.ceil((hold_seconds+30)/measured))
