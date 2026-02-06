import new_import_system
new_import_system.install(__file__)

from lele.Path import P
from lele.Metaprogramming import isinstance # bad idea ;; context switching
from lele.Metaprogramming import get_type_from_lazy_module as type # bad idea ;; context switching
