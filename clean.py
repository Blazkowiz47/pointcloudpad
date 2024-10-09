import os

for model in os.listdir("tmp/"):
    if "." in model:
        continue
    for iphone in ["iPhone11", "iPhone12"]:
        for i in range(1, 7):
            attack = f"Attack_{i}"
            #             if not os.path.isdir(os.path.join("tmp", model, iphone, attack)):
            #                 print(model, iphone, attack)
            #             continue

            dir = os.path.join("tmp", model, iphone, attack, "checkpoints")

            executions = os.listdir(dir)
            for execution in executions:
                if not os.listdir(os.path.join(dir, execution)):
                    os.system(f"rm -rf {os.path.join(dir, execution)}")
            dir = os.path.join("tmp", model, iphone, attack, "checkpoints")
            if not os.listdir(dir):
                os.system(f"rm -rf {os.path.join('tmp', model, iphone, attack)}")
