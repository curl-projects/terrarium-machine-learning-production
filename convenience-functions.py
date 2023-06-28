import modal
from pydantic import BaseModel

stub = modal.Stub("terrarium-machine-learning")


delete_prisma_data_image = modal.Image.debian_slim().pip_install("prisma")
@stub.function(mounts=[modal.Mount.from_local_dir("./prisma", remote_path="/root/prisma")],
               secret=modal.Secret.from_name("terrarium-secrets"),
               image=delete_prisma_data_image)
async def delete_prisma_frs():
    import subprocess

    subprocess.run('prisma generate', shell=True)

    from prisma import Prisma

    db = Prisma()
    await db.connect()

    response = await db.featurerequest.delete_many()

    print("Response:", response)

    await db.disconnect()

@stub.function(mounts=[modal.Mount.from_local_dir("./prisma", remote_path="/root/prisma")],
               secret=modal.Secret.from_name("terrarium-secrets"),
               image=delete_prisma_data_image)
async def delete_prisma_features():
    import subprocess

    subprocess.run('prisma generate', shell=True)

    from prisma import Prisma

    db = Prisma()
    await db.connect()

    response = await db.feature.delete_many()

    print("Response:", response)

    await db.disconnect()

@stub.function(mounts=[modal.Mount.from_local_dir("./prisma", remote_path="/root/prisma")],
               secret=modal.Secret.from_name("terrarium-secrets"),
               image=delete_prisma_data_image)
async def delete_prisma_datasets():
    import subprocess

    subprocess.run('prisma generate', shell=True)

    from prisma import Prisma

    db = Prisma()
    await db.connect()

    response = await db.dataset.delete_many()

    print("Response:", response)

    await db.disconnect()


