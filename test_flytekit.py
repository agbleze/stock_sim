
from flytekit import task, workflow

@task
def hello() -> str:
    return "Hello, World"

@workflow
def hello_wf() -> str:
    res = hello()
    return res

if __name__ == "__main__":
    print(f"running flyte code {hello_wf()}")