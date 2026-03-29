import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/junghun/dream_ws/neuro_ws/install/neuro_decision'
