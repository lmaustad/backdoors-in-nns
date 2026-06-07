# Auto-import all adapters to trigger @register_adapter decorators
from . import dfba_adapter  # noqa: F401
from . import arch_backdoors_adapter  # noqa: F401
from . import boone_bane_adapter  # noqa: F401
from . import hiding_needles_adapter  # noqa: F401
from . import trojannet_adapter  # noqa: F401
from . import foobar_adapter  # noqa: F401
from . import baseline_resnet_adapter  # noqa: F401
from . import model_editing_clip_adapter  # noqa: F401
from . import handcrafted_adapter  # noqa: F401
from . import badnets_adapter  # noqa: F401
from . import baseline_inception_adapter  # noqa: F401