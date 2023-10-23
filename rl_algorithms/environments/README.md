```base_env.py``` is the main environment strip.
```custom_env_0.py``` is an example of custom environment, to use these custom environments, you can initialize you environment like:

```python
from rl_algorithms.environments.env import *
from rl_algorithms.common.core import *

logger = Logger('./log.txt')
custom_env = Env('custom_env_0', logger)
```