import subprocess
import time
def addAllToGit(message:str):
    addf = "git add *"
    commit = f'git commit -m "{message}"'
    push = "git push"
    subprocess.call(addf)
    subprocess.call(commit)
    subprocess.call(push)
    
if __name__ == "__main__":
    for i in range(4, 1000):
        time.sleep(14400)
        addAllToGit(f"{i}")
