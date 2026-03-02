import subprocess as sp

p = sp.run(
    "ps -ef | grep aria | grep -v grep",
    shell=True,
)
print(p.stdout.decode())
