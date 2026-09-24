from core.router import route
from core.tools import calculator
assert route("hello").mode=="fast"
assert calculator("17*23")["result"]==391
print("smoke: OK")
