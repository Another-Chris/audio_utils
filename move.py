import os

def create_dir(labels):
    root = "./data"
    train = f"{root}/train"
    valid = f"{root}/valid"
    test = f"{root}/test"

    os.mkdir(root)
    os.mkdir(train)
    os.mkdir(valid)
    os.mkdir(test)
    for label in set(labels):
        os.mkdir(f"{train}/{label}")
        os.mkdir(f"{valid}/{label}")
        os.mkdir(f"{test}/{label}")


def create_folder_dir(root_name):
    root = f"./{root_name}"
    train = f"{root}/train"
    valid = f"{root}/valid"
    test = f"{root}/test"
    try:
        os.mkdir(root)
        os.mkdir(train)
        os.mkdir(valid)
        os.mkdir(test)
    except:
        print("folder already exists")
