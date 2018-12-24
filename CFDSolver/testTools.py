from Tools import *

a = [2, 2]
b = [1, 1, 1]
c = [3, 3]
x = [0, 0, 0]
f = [7, 13, 7]
try:
    assert [1, 2, 3] == list(map(round, solveTDM(a, b, c, x, f)))
except:
    print(list(map(round, solveTDM(a, b, c, x, f))))

    raise AssertionError("Failed normal matrix")

a = [2, 2]
b = [1, 1, 1]
c = [3, 3]
x = [1, 0, 3]
f = [7, 13, 7]
try:
    assert [1, 2, 3] == list(map(round, solveTDM(a, b, c, x, f)))
except:
    print(list(map(round, solveTDM(a, b, c, x, f))))
    raise AssertionError("Failed test with BCs")

# If we got here then it seems to work
print("solveTDM tests successful")
