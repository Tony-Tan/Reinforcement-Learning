```base_env.py``` is the main environment strip.
```custom_env_0.py``` is an example of custom environment, to use these custom environments, you can initialize you environment like:

```python
from environments.env_wrapper import *
from core import *

logger = Logger('./log.txt')
custom_env = EnvWrapper('custom_env_0', logger)
```