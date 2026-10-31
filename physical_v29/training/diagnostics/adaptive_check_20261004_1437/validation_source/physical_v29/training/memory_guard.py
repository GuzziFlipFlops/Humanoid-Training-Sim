"""Host-memory maintenance at completed PPO updates, separate from VRAM OOM.

Windows process commit grew to 67 GiB in a 32 GiB machine during r13. Recycle
the owned worker from its saved actor/critic/Adam before paging dominates.
"""


def recycle_reason(memory):
    if memory.get('ram_available_mb',100000) < 2048:
        return 'host_ram_below_2048MiB'
    # Commit includes driver/allocator reservations. This bound is deliberately
    # slightly above physical RAM, but prevents the observed unbounded growth.
    budget=max(12288,1.15*memory.get('ram_total_mb',100000))
    if memory.get('process_private_mb',0) > budget:
        return 'worker_commit_above_host_budget'
    return None


class ResourceRecycle(Exception):
    def __init__(self,iteration,reason,memory):
        super().__init__(reason)
        self.iterations=iteration+1
        self.reason=reason
        self.memory=memory


def restored_learning_rate(optimizer, ceiling):
    """Continue adaptive PPO at its saved rate after a worker restart."""
    rate=min(float(group['lr']) for group in optimizer.param_groups)
    rate=min(rate,ceiling)
    for group in optimizer.param_groups:group['lr']=rate
    return rate
