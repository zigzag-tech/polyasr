"""Test-wide guard: no test process may attach to a live Livestack broker.

Importing a server module is already inert (see test_import_isolation.py).
These settings are a second line of defence: they cover anything that calls
attach() in-process, on a host whose live broker (127.0.0.1:8799) and live
engine (127.0.0.1:8766) would otherwise answer. Port 9 (discard) has nothing
listening. test_import_isolation.py overrides these on purpose, and only in
its child processes.
"""
import os

os.environ["LIVESTACK_REGISTER"] = "0"
os.environ["LIVESTACK_MESH_ENABLED"] = "0"
os.environ["LIVESTACK_BROKER_URL"] = "http://127.0.0.1:9"
os.environ.pop("LIVESTACK_NODE_HOST", None)
