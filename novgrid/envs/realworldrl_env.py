from typing import Any, Dict, Optional
from shimmy.dm_control_compatibility import DmControlCompatibilityV0
import realworldrl_suite.envionments as rwrl

class RealWorldRLWrapper(DmControlCompatibilityV0):

    def __init__(self, domain_name: str='cartpole',
    task_name: str='realworld_swingup',
    combined_challenge: str='easy', render_mode: Optional[str] = None, render_kwargs: Optional[Dict[str, Any]] = None):
        env = rwrl.load(domain_name=domain_name, task_name=task_name, combined_challenge=combined_challenge)
        super().__init__(env, render_mode, render_kwargs)
