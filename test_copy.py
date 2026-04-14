import copy

import ml_peg.app.utils.utils as mpu
c1 = mpu.load_framework_registry()
c2 = mpu.load_framework_registry()
print(id(c1), id(c2))
c1["xyz"] = "abc"
print(c2)
